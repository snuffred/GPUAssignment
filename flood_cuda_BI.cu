/*
 * Simulation of rainwater flooding / 雨水洪涝模拟
 * CUDA version - optimized / CUDA 版本 - 已优化
 *
 * Changes vs the baseline parallel version:
 * 相比基础并行版本的改动：
 *   (1) Rainfall: one kernel launch per cloud, grid sized to the cloud's
 *       bounding box -> threads never idle-iterate over other clouds.
 *       降雨：每朵云启动一个 kernel，grid 尺寸与云的包围盒一致，
 *       线程不再需要空转遍历其他云。
 *   (2) Spillage: gather-based. Compute kernel writes per-cell
 *       outflow_level + outflow_proportion + cell_height. Apply kernel
 *       reads its own outflow and gathers inflow from 4 neighbours.
 *       Eliminates the rows*columns*4 scatter scratch buffer.
 *       溢流：采用 gather（收集）方式。compute kernel 为每个单元写入
 *       outflow_level、outflow_proportion 和 cell_height。apply kernel
 *       读取自己的 outflow 并从 4 个邻居处收集 inflow，
 *       从而省去 rows*columns*4 大小的 scatter 暂存缓冲区。
 *   (3) Per-minute max_spillage readback uses pinned memory + async copy,
 *       and the apply kernel is queued before the copy so it overlaps.
 *       Final totals (total_water, max_water) are reduced on the GPU so
 *       we don't need to copy the whole water_level matrix back.
 *       每分钟的 max_spillage 回读使用 pinned 内存 + 异步拷贝，
 *       apply kernel 在拷贝之前入队以便两者重叠。
 *       最终的统计量 (total_water, max_water) 在 GPU 上完成 reduce，
 *       因此无需将整个 water_level 矩阵拷回主机。
 */

/* ==================================================================
 * HEADER INCLUDES / 头文件
 * ================================================================== */
#include <float.h>   // For DBL_MAX / 用于 DBL_MAX，作为 max_spillage_iter 的初始哨兵值
#include <math.h>    // sqrtf / cosf / sinf / ceilf / fmaxf / M_PI 等数学函数
#include <stdio.h>   // fprintf / stderr，用于错误输出
#include <stdlib.h>  // malloc / free / exit / EXIT_FAILURE，主机端内存管理
#include <string.h>  // memcpy，用于浮点数与整数的位级别重解释

#include <cuda.h>    // CUDA runtime API：cudaMalloc / cudaMemcpy / kernel 启动语法等

/* ==================================================================
 * ERROR-CHECKING MACROS / 错误检查宏
 * ==================================================================
 * These macros wrap CUDA API calls and kernel launches to check for
 * errors without cluttering the main logic.
 * 这些宏包装 CUDA API 调用和 kernel 启动以检查错误，
 * 同时不污染主逻辑。出错时会打印出错行号和 CUDA 的错误字符串。
 * ================================================================== */

/* CUDA_CHECK_FUNCTION wraps an API call that returns cudaError_t.
 * 该宏包装返回 cudaError_t 的 CUDA API 调用。
 * 示例 Example: CUDA_CHECK_FUNCTION(cudaMalloc(&p, n));
 * 把返回值存入 check，若非 success 则打印错误。 */
#define CUDA_CHECK_FUNCTION(call)                                                                 \
    {                                                                                             \
        cudaError_t check = call;                                                                 \
        if (check != cudaSuccess)                                                                 \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

/* CUDA_CHECK_KERNEL polls the last error via cudaGetLastError().
 * 该宏通过 cudaGetLastError() 查询最近一次的错误。
 * kernel 启动是异步的且使用 <<<>>> 语法，错误以"粘性"状态
 * 保留，而不是直接作为返回值返回。 */
#define CUDA_CHECK_KERNEL()                                                                              \
    {                                                                                                    \
        cudaError_t check = cudaGetLastError();                                                          \
        if (check != cudaSuccess)                                                                        \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

#include "rng.c"   // Random number generator / 框架提供的随机数生成器
#include "flood.h" // Problem parameters, macros / 问题参数、结果结构体、FIXED/FLOATING/accessMat 等宏

/* get_time() is defined in a C file; extern "C" avoids C++ name mangling.
 * get_time() 定义在 C 文件中；extern "C" 可避免 C++ 的符号修饰，
 * 这样链接器才能找到该函数。 */
extern "C" double get_time();

/* ==================================================================
 * THREAD-BLOCK GEOMETRY / 线程块几何
 * ==================================================================
 * We use 16x16 = 256 threads per block. This is a sweet spot for most
 * GPUs: it matches a whole warp count (8 warps), fits nicely in
 * shared memory and registers.
 * 每个 block 使用 16x16 = 256 个线程。这对大多数 GPU 是最佳点：
 * 正好 8 个 warp，共享内存和寄存器占用都较为合适，
 * 并且给出良好的二维 tile 划分。 */
#define BLOCK_X 16                          // Threads per block along X / 每块 X 方向线程数
#define BLOCK_Y 16                          // Threads per block along Y / 每块 Y 方向线程数
#define BLOCK_THREADS (BLOCK_X * BLOCK_Y)   // Total threads per block=256 / 每块总线程数

/* ==================================================================
 * BLOCK-LEVEL REDUCTIONS / 块级 reduce
 * ==================================================================
 * Each kernel produces per-thread contributions. Summing/maxing them
 * at the block level first reduces the number of expensive atomic
 * operations from "one per thread" to "one per block".
 * 每个 kernel 都会产生每线程的贡献（降雨量、水分流失、最大溢流等）。
 * 先在 block 级别做 sum/max，再通过 atomic 操作写到全局内存，
 * 可把昂贵的原子操作数从"每线程一次"降到"每块一次"。 */

/* Sum reduction of unsigned long long across all threads of a block.
 * Result is atomically added to *global_sum.
 * 对一个 block 内全部线程的 unsigned long long 值做求和 reduce，
 * 结果原子地累加到 *global_sum 所指向的全局计数器。
 * 使用共享内存作为 block 的临时缓冲区。 */
__device__ static inline void block_reduce_add_ull(
    unsigned long long value, unsigned long long *global_sum)
{
    // One slot of shared memory per thread / 每线程一个共享内存槽位。
    __shared__ unsigned long long shared_data[BLOCK_THREADS];

    // Flatten the 2D thread index into 1D id / 把 2D 线程索引展平为 1D ID，范围 [0, BLOCK_THREADS)。
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread writes its own value / 每个线程写入自己的贡献值。
    shared_data[threadId] = value;

    // Sync before reduction starts / 开始 reduce 前必须让所有写入对全体线程可见。
    __syncthreads();

    // Classic tree-based reduction / 经典的树形 reduce：
    // Halve stride each round / 每轮把 stride 减半，
    // top half accumulates from bottom half / 上半部分累加下半部分的值，
    // until shared_data[0] holds the block total / 直到 shared_data[0] 保存整块的总和。
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
            shared_data[threadId] += shared_data[threadId + stride];
        __syncthreads();  // Barrier between tree levels / 树形各层之间的屏障。
    }

    // Only thread 0 does the global atomic / 只有线程 0 执行全局原子加。
    // Skip when total is zero to avoid contention / 当整块合计为 0 时跳过 atomic，减少竞争。
    if (threadId == 0 && shared_data[0] != 0ULL)
        atomicAdd(global_sum, shared_data[0]);
}

/* Max reduction for a non-negative float across the block, stored as
 * int bits. For non-negative IEEE-754 floats, comparing raw bit patterns
 * as signed ints gives the same ordering as comparing them as floats.
 * 对一个 block 内所有线程的非负 float 做 max reduce，
 * 结果以 int bit 形式写入 *global_max（atomicMax 只支持 int，
 * 因此我们用 __float_as_int() 将 float 的位模式重解释为 int）。
 * 对非负 IEEE-754 浮点数来说，以有符号整数比较其位模式
 * 与以 float 比较得到的顺序是一致的，该技巧成立。 */
__device__ static inline void block_reduce_max_float_nooneg(
    float value, int *global_max)
{
    __shared__ float shared_data[BLOCK_THREADS];
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    shared_data[threadId] = value;
    __syncthreads();

    // Tree-based max reduction / 树形 max reduce。
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
        {
            float other = shared_data[threadId + stride];
            if (other > shared_data[threadId])
                shared_data[threadId] = other;
        }
        __syncthreads();
    }

    // Only thread 0 writes, and only if > 0 / 只有线程 0 做 atomic，且仅在 max > 0 时执行。
    // Many blocks have no activity in a given minute, so skipping zeros
    // is a worthwhile optimization.
    // 许多 block 在一分钟内没有降雨/溢流，跳过零值是有价值的优化。
    if (threadId == 0 && shared_data[0] > 0.0f)
        atomicMax(global_max, __float_as_int(shared_data[0]));
}

/* ==================================================================
 * CUDA KERNELS / CUDA 核函数
 * ================================================================== */

/* ------------------------------------------------------------------
 * kernel_rain_cloud / 降雨 kernel（单朵云）
 * ------------------------------------------------------------------
 * Rainfall from a single cloud. CPU launches one kernel per cloud,
 * sized to the cloud's bounding box, so threads never waste time
 * testing cells outside any cloud's radius.
 * 单朵云的降雨计算。CPU 对每朵云启动一个 kernel，
 * grid 大小等于云的包围盒，线程不会浪费时间去判定
 * 不在任何云半径内的单元格，也不会遍历一个云列表。
 *
 * Parameters / 参数：
 *   water_level      - 2D grid of fixed-point water / 定点整数格式的水量矩阵
 *   cloud            - cloud position/intensity/radius (by value) / 云的信息（值传递）
 *   row_start,col_start - float top-left of bounding box / 包围盒左上角（浮点，亚格精度）
 *   row_first,col_first - integer top-left of bounding box / 包围盒左上角的整数索引
 *   K_rows,K_cols    - bounding box dimensions / 包围盒尺寸（以单元格数计）
 *   ex_factor        - global rainfall scaling factor / 全局降雨强度缩放因子
 *   rows,columns     - full grid dimensions / 整个矩阵的行列数（供 accessMat 使用）
 *   d_total_rain     - global accumulator for rainfall / 全局累计降雨量（定点）
 * ------------------------------------------------------------------ */
__global__ void kernel_rain_cloud(
    int *__restrict__ water_level,      // __restrict__ 提示编译器无别名 / no-alias hint
    Cloud_t cloud,                       // Pass by value -> param memory / 值传递，保存在参数/常量内存
    float row_start, float col_start,
    int row_first, int col_first,
    int K_rows, int K_cols,
    float ex_factor,
    int rows, int columns,
    unsigned long long *d_total_rain)
{
    // Thread's position within the bounding box / 线程在包围盒内的 (row, col) 位置。
    int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    // This thread's rain contribution; stays 0 if outside the cloud.
    // 本线程的降雨贡献；若单元格在云半径外或在包围盒外，则保持为 0。
    int added_fixed = 0;

    // Bounding-box guard: grid may be slightly larger than the box due to
    // block-size rounding. Threads outside still join the block reduction.
    // 包围盒范围判断：grid 可能因 block 大小向上取整而略大于包围盒。
    // 越界线程什么都不做，但仍要参加块内 reduce（贡献 0）。
    if (local_row < K_rows && local_col < K_cols)
    {
        // Local box index -> absolute matrix index / 把局部包围盒索引转为全局矩阵索引。
        int row_pos = row_first + local_row;
        int col_pos = col_first + local_col;

        // Sub-cell-precision matrix coords as floats / 亚格精度的浮点矩阵坐标。
        float row_pos_f = row_start + (float)local_row;
        float col_pos_f = col_start + (float)local_col;

        // Matrix -> scenario (km) coordinates / 把矩阵坐标转换为真实的场景 (km) 坐标。
        float x_pos = COORD_MAT2SCEN_X(col_pos_f);
        float y_pos = COORD_MAT2SCEN_Y(row_pos_f);

        // Distance from this cell to the cloud centre in km.
        // 此单元到云中心的距离（单位 km）。
        float dx = x_pos - cloud.x;
        float dy = y_pos - cloud.y;
        float distance = sqrtf(dx * dx + dy * dy);

        // Only cells inside the radius receive rain / 只有云半径范围内的格子会接雨。
        if (distance < cloud.radius)
        {
            // Rain intensity falls off linearly with normalized distance.
            // fmaxf(0, ...) guards against edge-underflow.
            // 降雨强度随归一化距离线性下降；
            // fmaxf(0, ...) 保证边缘处不会出现负值。
            float rain = ex_factor *
                         fmaxf(0.0f, cloud.intensity - distance / cloud.radius * sqrtf(cloud.intensity));

            // mm/hour -> meters/minute for accumulation.
            // 把 mm/h 转换为 m/min 以便累加。
            float meters_per_minute = rain / 1000.0f / 60.0f;

            // float meters -> fixed-point int (micro-meters).
            // 把浮点(米) 转为定点整数 (微米级) 以避免浮点累加误差。
            added_fixed = FIXED(meters_per_minute);

            // Write rain to water_level; skip zero for a tiny saving.
            // 把降雨写入 water_level；增量为 0 时跳过以省掉一次写入。
            if (added_fixed != 0)
                accessMat(water_level, row_pos, col_pos) += added_fixed;
        }
    }

    // All threads join the reduction (syncthreads inside requires it).
    // 所有线程都必须参加 reduce，因为内部的 __syncthreads 要求
    // 整块的所有线程都到达该点。
    block_reduce_add_ull((unsigned long long)added_fixed, d_total_rain);
}

/* ------------------------------------------------------------------
 * kernel_spillage_compute / 溢流计算 kernel
 * ------------------------------------------------------------------
 * First half of spillage. For each cell computes:
 * 溢流步骤的前半部分。对每个单元格计算：
 *   outflow_level       - water amount this cell will shed
 *                       - 该单元格将排出的水量
 *   outflow_proportion  - proportional redistribution factor
 *                       - 比例分配因子
 *   cell_height         - ground + water (cached so apply kernel
 *                          doesn't re-read water_level while it's being updated)
 *                       - 地面+水位（缓存，这样 apply kernel 就不必
 *                          在 water_level 被就地更新时再次读取它）
 *
 * Also accumulates / 同时累加：
 *   d_total_loss     - fixed-point total water lost through borders
 *                    - 从网格边界流失的水量（定点）
 *   d_max_spill_bits - global max outflow_level (int-bits encoded)
 *                    - 全局最大 outflow_level（以 int bit 编码）
 *
 * Splitting into two kernels makes spillage race-free: in scatter
 * implementations, apply would write directly into neighbour cells
 * (data race). Here each thread only updates its own water_level,
 * using frozen neighbour state from compute -> no atomics needed.
 * 拆分成两个 kernel 的关键是实现无竞争：在 scatter 实现中，
 * apply 会直接写入邻居单元格 (数据竞争)；而这里每个线程
 * 只更新自己的 water_level，所需的邻居信息来自 compute
 * 中写入的"冻结"缓冲区，因此不需要原子写操作。
 * ------------------------------------------------------------------ */
__global__ void kernel_spillage_compute(
    const int *__restrict__ water_level,       // READ-ONLY here / 本 kernel 中只读
    const float *__restrict__ ground,           // Terrain heights (static) / 地形高度（静态）
    float *__restrict__ outflow_level,          // OUTPUT: per-cell outflow / 输出：每格溢流量
    float *__restrict__ outflow_proportion,     // OUTPUT: redistribution factor / 输出：比例因子
    float *__restrict__ cell_height,            // OUTPUT: ground + water / 输出：地面+水位
    unsigned long long *d_total_loss,           // Reduction: boundary loss / 边界流失累加
    int *d_max_spill_bits,                      // Reduction: max outflow / 最大 outflow
    int rows, int columns)
{
    // Absolute grid coords for this thread / 本线程的全局网格坐标。
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // Per-thread scratch values / 每线程的临时值。默认值即为越界线程的贡献。
    long local_loss = 0;                 // This cell's boundary loss contribution / 本格的边界流失
    float local_spillage_level = 0.0f;   // This cell's outflow amount / 本格的溢流量
    float computed_proportion = 0.0f;    // This cell's redistribution factor / 本格的分配因子

    // In-bounds guard / 越界保护。
    if (row_pos < rows && col_pos < columns)
    {
        // Read own water (fixed-point int) and ground (float).
        // 读取本格的水量（定点整数）和地面高度（float）。
        int w = accessMat(water_level, row_pos, col_pos);
        float ground_self = accessMat(ground, row_pos, col_pos);
        float water_f = FLOATING(w);                   // int -> float meters / 定点转浮点(米)
        float current_height = ground_self + water_f; // Total surface height / 总表面高度

        // Only cells with water can spill / 只有有水的单元格才会溢出。
        if (w > 0)
        {
            float sum_diff = 0.0f;  // Total height drop to "lower" neighbours / 到所有低邻居的高差之和。

            // Loop over 4 neighbours / 遍历上下左右 4 个邻居。
            // #pragma unroll fully unrolls a known-length loop for better scheduling.
            // #pragma unroll 提示编译器完全展开这个已知长度的循环，
            // 利于指令调度。
#pragma unroll
            for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
            {
                int nr = row_pos + displacements[cp][0];
                int nc = col_pos + displacements[cp][1];
                float nh;
                // Boundary handling: treat out-of-grid neighbours as having
                // the same ground height but zero water, so water "falls off".
                // 边界处理：越界的邻居视为地面高度与本格相同但水量为 0，
                // 这样边界处的水会"掉出"网格。
                if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
                    nh = ground_self;
                else
                    nh = accessMat(ground, nr, nc) +
                         FLOATING(accessMat(water_level, nr, nc));

                // Only neighbours at/below us receive spill / 只有高度不高于本格的邻居才接收溢流。
                if (current_height >= nh)
                {
                    float hd = current_height - nh;
                    sum_diff += hd;
                    // The maximum height drop becomes our spillage level
                    // (later capped by available water).
                    // 最大高差成为本格的溢流水位
                    // （随后会被可用水量限制上限）。
                    if (hd > local_spillage_level)
                        local_spillage_level = hd;
                }
            }

            // Can't spill more water than we have / 溢出不能超过本格现有水量。
            if (water_f < local_spillage_level)
                local_spillage_level = water_f;

            // If there are downhill neighbours, compute proportion.
            // 若存在下坡的邻居，则计算比例因子。
            if (sum_diff > 0.0f)
            {
                float prop = local_spillage_level / sum_diff;
                // Tiny-proportion filter avoids denormal/near-zero noise
                // and ensures apply's tests remain meaningful.
                // 过滤极小的 prop 以避免次正规数和近零噪声，
                // 同时保证后续 apply 里的比较具有意义。
                if (prop > 1e-8f)
                {
                    computed_proportion = prop;
                    // Count boundary losses / 统计从边界流失的水量：
                    // 每个越界"邻居"贡献 (prop * 高差 / 2)。
#pragma unroll
                    for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
                    {
                        int nr = row_pos + displacements[cp][0];
                        int nc = col_pos + displacements[cp][1];
                        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
                        {
                            if (current_height >= ground_self)
                                local_loss += FIXED(prop * (current_height - ground_self) / 2);
                        }
                    }
                }
                else
                {
                    // Too small: pretend we spill nothing / 比例太小：视作不溢流。
                    local_spillage_level = 0.0f;
                }
            }
            else
            {
                // No downhill neighbour / 没有下坡邻居：无水可溢出。
                local_spillage_level = 0.0f;
            }
        }

        // Write the three outputs / 写入三个输出缓冲区。
        accessMat(outflow_level, row_pos, col_pos) = local_spillage_level;
        accessMat(outflow_proportion, row_pos, col_pos) = computed_proportion;
        accessMat(cell_height, row_pos, col_pos) = current_height;
    }

    // Block-level reductions / 块级 reduce。
    // All threads in the block must reach these calls.
    // 同一 block 的全部线程都必须到达这些调用。
    block_reduce_add_ull((unsigned long long)local_loss, d_total_loss);
    block_reduce_max_float_nooneg(local_spillage_level, d_max_spill_bits);
}

/* ------------------------------------------------------------------
 * kernel_spillage_apply / 溢流应用 kernel
 * ------------------------------------------------------------------
 * GATHER-based second half of spillage:
 * 溢流步骤的后半部分，采用 gather 方式：
 *   delta = -own_outflow/SPILLAGE_FACTOR
 *         + sum_{neighbours}(n_prop * (n_height - my_height))/SPILLAGE_FACTOR
 * Each cell writes ONLY to its own water_level. No atomicAdd, no race
 * despite every thread reading 4 neighbours' state, because those reads
 * come from static compute outputs.
 * 每个单元格只写它自己的 water_level 位置，因此不需要 atomicAdd，
 * 也不会出现竞争：虽然每个线程都会读取 4 个邻居的状态，
 * 但这些读操作读的是 compute 阶段产生的静态缓冲区。
 * ------------------------------------------------------------------ */
__global__ void kernel_spillage_apply(
    int *__restrict__ water_level,                // WRITE / 写：更新的水量
    const float *__restrict__ cell_height,         // Frozen total heights / 冻结的总高度
    const float *__restrict__ outflow_level,       // Own outflow / 本格的 outflow
    const float *__restrict__ outflow_proportion,  // Neighbour inflow factor / 邻居的 inflow 因子
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // Early-out: no reduction here, so out-of-range threads can return.
    // 提前返回：此 kernel 不做 reduce，因此越界线程可直接返回。
    if (row_pos >= rows || col_pos >= columns)
        return;

    // Pull own height/outflow from cached buffers / 从缓存缓冲区读取本格的高度与 outflow。
    float my_height = accessMat(cell_height, row_pos, col_pos);
    float my_outflow = accessMat(outflow_level, row_pos, col_pos);

    // Net change in fixed-point units, starting at 0.
    // 以定点单位记录净变化，初始为 0。
    int delta = 0;

    // Step 1: subtract own outflow (divided by SPILLAGE_FACTOR).
    // 第 1 步：扣除自己的 outflow（除以 SPILLAGE_FACTOR）。
    if (my_outflow > 0.0f)
        delta -= FIXED(my_outflow / SPILLAGE_FACTOR);

    // Step 2: add inflow from each of 4 neighbours.
    // 第 2 步：加上 4 个邻居流入的水量。
#pragma unroll
    for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
    {
        int nr = row_pos + displacements[cp][0];
        int nc = col_pos + displacements[cp][1];
        float infl = 0.0f;
        // In-bounds neighbours only / 只考虑在网格内的邻居。
        if (nr >= 0 && nr < rows && nc >= 0 && nc < columns)
        {
            float n_prop = accessMat(outflow_proportion, nr, nc);
            if (n_prop > 0.0f)
            {
                float nh = accessMat(cell_height, nr, nc);
                // Only receive from higher neighbours (water flows downhill).
                // 只从比本格更高的邻居那里接收水（水往低处流）。
                if (nh >= my_height)
                    infl = n_prop * (nh - my_height);
            }
        }
        delta += FIXED(infl / SPILLAGE_FACTOR);
    }

    // Single write per cell; no atomic needed.
    // 每个单元只有一次写入；无需原子操作。
    if (delta != 0)
        accessMat(water_level, row_pos, col_pos) += delta;
}

/* ------------------------------------------------------------------
 * kernel_final_stats / 最终统计 kernel
 * ------------------------------------------------------------------
 * Runs once after the main loop to compute total_water and
 * max_water_scenario on the GPU, avoiding a full water_level copyback.
 * Sum+max in a single pass using two shared-memory arrays.
 * 主循环结束后只运行一次，在 GPU 上计算 total_water 和
 * max_water_scenario，避免把整个 water_level 回拷到主机。
 * 通过两个共享内存数组在一次遍历中同时完成 sum + max。
 * ------------------------------------------------------------------ */
__global__ void kernel_final_stats(
    const int *__restrict__ water_level,
    unsigned long long *d_total_water,    // Sum of all fixed-point waters / 全部定点水量的总和
    int *d_max_water_bits,                 // Max water (float bits) / 最大水量（以 float 位形式存）
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    long long local_water = 0;    // Per-thread sum contribution / 每线程的 sum 贡献
    float local_max_f = 0.0f;     // Per-thread max contribution / 每线程的 max 贡献
    if (row_pos < rows && col_pos < columns)
    {
        int w = accessMat(water_level, row_pos, col_pos);
        local_water = w;               // Accumulate raw fixed-point / 累加原始定点值
        local_max_f = FLOATING(w);     // Convert to float for max / 转为 float 做 max
    }

    // Two shared buffers -> one tree reduction for sum AND max.
    // 两个共享缓冲区：一次树形 reduce 同时完成 sum 与 max。
    __shared__ long long sh_sum[BLOCK_THREADS];
    __shared__ float sh_max[BLOCK_THREADS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sh_sum[tid] = local_water;
    sh_max[tid] = local_max_f;
    __syncthreads();

    // Combined tree reduction / 合并的树形 reduce：每轮把活跃集合缩半。
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sh_sum[tid] += sh_sum[tid + s];
            if (sh_max[tid + s] > sh_max[tid])
                sh_max[tid] = sh_max[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes to the global atomics / 每个 block 的线程 0 执行全局原子操作。
    if (tid == 0)
    {
        atomicAdd(d_total_water, (unsigned long long)sh_sum[0]);
        if (sh_max[0] > 0.0f)
            atomicMax(d_max_water_bits, __float_as_int(sh_max[0]));
    }
}

/* ==================================================================
 * HOST-SIDE DRIVER: do_compute / 主机端驱动函数
 * ==================================================================
 * Called by the framework. Steps:
 * 框架调用的入口函数。执行步骤：
 *   1. Allocate device memory / 分配设备内存
 *   2. Main simulation loop (rain + spillage) for up to num_minutes,
 *      or until max_spillage drops below threshold.
 *      主模拟循环（降雨 + 溢流），最多运行 num_minutes 分钟，
 *      或直到 max_spillage 低于阈值。
 *   3. Final statistics / 计算最终统计量。
 *   4. Free everything / 清理所有资源。
 * ================================================================== */
extern "C" void
do_compute(struct parameters *p, struct results *r)
{
    // Cache grid dimensions and iteration-counter pointer.
    // 缓存网格尺寸和迭代计数器指针。
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    // Pick GPU 0 and wait for it to be idle before we start timing.
    // 选择 0 号 GPU，并在开始计时前等待其空闲。
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    // Total cells; size_t handles rows*columns > INT_MAX.
    // 总单元数；使用 size_t 以防溢出 int。
    size_t n_cells = (size_t)rows * (size_t)columns;

    /* ------------ Device pointers / 设备端指针 ------------ */
    int *d_water_level;                                                      // Fixed-point water / 定点水量
    float *d_ground, *d_outflow_level, *d_outflow_proportion, *d_cell_height; // 4 float grids / 四个 float 矩阵
    unsigned long long *d_total_rain, *d_total_loss, *d_total_water;         // Global counters / 全局计数器
    int *d_max_spill_bits, *d_max_water_bits;                                // Float-bits max counters / 以 float bits 存的最大值

    /* ------------ Allocate device memory / 分配设备内存 ------------ */
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_water_level, sizeof(int) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_ground, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_outflow_level, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_outflow_proportion, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_cell_height, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_rain, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_loss, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_water, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spill_bits, sizeof(int)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_water_bits, sizeof(int)));

    /* ------------ Zero-initialise all grids and counters
     *              所有矩阵和计数器清零 ------------
     * cudaMemset is a fast device-side bulk zeroing.
     * Note: we don't zero d_max_spill_bits here because it's reset each minute.
     * cudaMemset 是快速的设备端批量清零；
     * 注意 d_max_spill_bits 不在这里清零，因为它在每分钟开头都会重置。 */
    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, sizeof(int) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_level, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_proportion, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_cell_height, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_rain, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_loss, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_water, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_max_water_bits, 0, sizeof(int)));

    // Copy static ground heights to device (one-shot).
    // 把静态的地面高度从主机拷到设备（一次性）。
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, p->ground, sizeof(float) * n_cells, cudaMemcpyHostToDevice));

    /* ------------ Pinned host buffer for max_spill readback
     *              max_spill 回读使用的 pinned 主机缓冲 ------------
     * Pinned (page-locked) memory enables true async DMA and avoids
     * a runtime staging step, letting the copy overlap with kernels.
     * Pinned（页锁定）内存支持真正的异步 DMA 拷贝，
     * 省去 runtime 的中转步骤，使拷贝能与 kernel 执行重叠。 */
    int *h_max_spill_pinned;
    CUDA_CHECK_FUNCTION(cudaHostAlloc(&h_max_spill_pinned, sizeof(int), cudaHostAllocDefault));

    // Loop-termination sentinel; start at "infinity" so first iter always runs.
    // 循环终止条件的哨兵值；初始为"无穷大"以保证首次迭代一定进入。
    double max_spillage_iter = DBL_MAX;

    // Thread-block shape and full-grid size / 线程块形状和覆盖全矩阵的 grid。
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_full((columns + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Start wall-clock timer / 启动墙钟计时。
    r->runtime = get_time();

    /* ==============================================================
     * MAIN SIMULATION LOOP / 主模拟循环
     * Runs until num_minutes reached, or max spillage < threshold.
     * 当 num_minutes 达到，或上一分钟的最大溢流低于阈值时停止。
     * ============================================================== */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++)
    {
        /* Step 1: move clouds (CPU, small work).
         * 第 1 步：在 CPU 上移动云（工作量很小）。
         * num_clouds is usually small (tens~hundreds), so updating on
         * the host avoids kernel-launch overhead for negligible work.
         * num_clouds 通常很小（几十到几百），放 CPU 避免 kernel
         * 启动开销，性价比更好。 */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            Cloud_t *c = &p->clouds[cloud];
            // km/h -> km/min / 把 km/h 的速度换算成 km/min。
            float km_minute = c->speed / 60.0f;
            // Advance along angle / 沿着角度方向推进云心。
            c->x += km_minute * cosf(c->angle * (float)M_PI / 180.0f);
            c->y += km_minute * sinf(c->angle * (float)M_PI / 180.0f);
        }

        /* Step 2: rainfall, one kernel per cloud over its bounding box.
         * 第 2 步：对每朵云在其包围盒内启动一个 kernel 做降雨。
         * Kernels go into the default stream and execute in order; threads
         * within each kernel are parallel.
         * 这些 kernel 进入默认 stream 顺序执行；单个 kernel 内部
         * 的线程仍然是完全并行的。 */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            // Copy cloud struct (pass by value into the kernel).
            // 拷贝 cloud 结构（以值传递进入 kernel）。
            Cloud_t c = p->clouds[cloud];

            // Axis-aligned bounding box in matrix coords, clipped to scenario.
            // Floats allow sub-cell precision when the cloud is not grid-aligned.
            // 在矩阵坐标系下的轴对齐包围盒，并裁剪到场景范围内；
            // 使用浮点可以在云未对齐网格时保留亚格精度。
            float row_start = COORD_SCEN2MAT_Y(MAX(0.0f, c.y - c.radius));
            float row_end   = COORD_SCEN2MAT_Y(MIN(c.y + c.radius, (float)SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0.0f, c.x - c.radius));
            float col_end   = COORD_SCEN2MAT_X(MIN(c.x + c.radius, (float)SCENARIO_SIZE));

            // Integer top-left and cell-count extents. ceilf avoids losing a
            // partial row compared to a truncating cast.
            // 整数左上角与单元数。使用 ceilf 比截断更安全，不会漏掉
            // 部分行/列。
            int row_first = (int)row_start;
            int col_first = (int)col_start;
            int K_rows = (int)ceilf(row_end - row_start);
            int K_cols = (int)ceilf(col_end - col_start);

            // Entirely off-grid -> skip / 云完全在网格外：跳过。
            if (K_rows <= 0 || K_cols <= 0)
                continue;

            // Clamp start to 0 (clouds can wander off-grid).
            // 把起点 clamp 到 0（云可能飘出网格边界）。
            if (row_first < 0)
                row_first = 0;
            if (col_first < 0)
                col_first = 0;
            // Clamp extent to grid size / 把范围 clamp 到实际网格大小。
            if (row_first + K_rows > rows)
                K_rows = rows - row_first;
            if (col_first + K_cols > columns)
                K_cols = columns - col_first;
            if (K_rows <= 0 || K_cols <= 0)
                continue;

            // Grid sized exactly to the clamped box / Grid 尺寸恰好匹配裁剪后的包围盒。
            dim3 grid_cloud((K_cols + block.x - 1) / block.x,
                            (K_rows + block.y - 1) / block.y);

            // Launch rainfall kernel for this single cloud.
            // 为这一朵云启动降雨 kernel。
            kernel_rain_cloud<<<grid_cloud, block>>>(
                d_water_level, c, row_start, col_start,
                row_first, col_first, K_rows, K_cols,
                p->ex_factor, rows, columns, d_total_rain);
        }

        /* Step 3: spillage compute.
         * 第 3 步：溢流 compute。
         * Reset max_spill to 0 (async, overlaps with previously queued work).
         * 把 max_spill 异步清零（与流中排队的前序工作重叠）。 */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spill_bits, 0, sizeof(int)));
        kernel_spillage_compute<<<grid_full, block>>>(
            d_water_level, d_ground,
            d_outflow_level, d_outflow_proportion, d_cell_height,
            d_total_loss, d_max_spill_bits, rows, columns);

        /* Step 4: spillage apply. Queued before the max_spill readback
         * so the copy can overlap with kernel execution.
         * 第 4 步：溢流 apply。在 max_spill 回读之前入队，
         * 这样拷贝可以与 kernel 执行并行。 */
        kernel_spillage_apply<<<grid_full, block>>>(
            d_water_level, d_cell_height,
            d_outflow_level, d_outflow_proportion, rows, columns);

        /* Step 5: async readback + synchronous wait (pinned host memory).
         * 第 5 步：异步回拷 + 同步等待（pinned 主机内存）。
         * cudaMemcpyAsync + pinned = true DMA that overlaps with apply.
         * Then stream-sync to ensure the value is ready before host reads.
         * cudaMemcpyAsync 配合 pinned 内存就是真正的 DMA，
         * 能与 apply 并行；随后用 stream sync 保证主机读取前数值已到位。 */
        CUDA_CHECK_FUNCTION(cudaMemcpyAsync(
            h_max_spill_pinned, d_max_spill_bits, sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK_FUNCTION(cudaStreamSynchronize(0));

        // Decode the max_spill value: stored as int-bits of a float.
        // 解码 max_spill：它以 float 的 int bits 方式存储。
        // 0 表示"本分钟没有溢流"。
        float max_sl;
        if (*h_max_spill_pinned == 0)
            max_sl = 0.0f;
        else
            memcpy(&max_sl, h_max_spill_pinned, sizeof(int));

        // Divide by SPILLAGE_FACTOR to get per-minute maximum.
        // 除以 SPILLAGE_FACTOR 得到本分钟真正的最大溢流量。
        max_spillage_iter = (double)max_sl / SPILLAGE_FACTOR;

        // Track overall max across the whole simulation.
        // 跟踪整个模拟过程的最大溢流。
        if (max_spillage_iter > r->max_spillage_scenario)
        {
            r->max_spillage_scenario = max_spillage_iter;
            r->max_spillage_minute = *minute;
        }
    }

    /* Final reduction on the GPU: total_water and max_water_scenario
     * without copying the whole matrix back.
     * 在 GPU 上完成最终 reduce：计算 total_water 与 max_water_scenario，
     * 无需把整个矩阵回拷到主机。 */
    kernel_final_stats<<<grid_full, block>>>(
        d_water_level, d_total_water, d_max_water_bits, rows, columns);

    // Pull the four scalar reductions back to host.
    // 把四个标量 reduce 结果回拷到主机。
    unsigned long long h_total_rain = 0, h_total_loss = 0, h_total_water = 0;
    int h_max_water_bits = 0;
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_rain, d_total_rain, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_loss, d_total_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_water, d_total_water, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_max_water_bits, d_max_water_bits, sizeof(int), cudaMemcpyDeviceToHost));

    // Wait for everything, then stop the timer.
    // 等待所有工作完成，然后停止计时。
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    r->runtime = get_time() - r->runtime;

    // Fill results struct. Unsigned -> signed cast for the framework.
    // 填写结果结构体。注意类型：我们用 unsigned 存储，框架用 signed long。
    r->total_rain += (long)h_total_rain;
    r->total_water_loss += (long)h_total_loss;
    r->total_water = (long)h_total_water;

    // Decode max_water_scenario (int-bits -> float).
    // 解码 max_water_scenario（int bits -> float）。
    if (h_max_water_bits == 0)
    {
        r->max_water_scenario = 0.0;
    }
    else
    {
        float maxf;
        memcpy(&maxf, &h_max_water_bits, sizeof(float));
        r->max_water_scenario = maxf;
    }

    /* Optional: dump the final water matrix if requested on the cmdline.
     * Developer/test feature only; production runs set final_matrix=0
     * and skip the big copy + print.
     * 可选：按命令行参数请求输出最终水位矩阵。
     * 仅开发/测试使用；正式运行设 final_matrix=0 即可跳过大拷贝与打印。 */
    if (p->final_matrix)
    {
        int *water_level = (int *)malloc(sizeof(int) * n_cells);
        if (!water_level)
        {
            fprintf(stderr, "Error: Failed to allocate memory for water level array\n");
            exit(EXIT_FAILURE);
        }
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, sizeof(int) * n_cells, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
        free(water_level);
    }

    /* ------------ Cleanup / 清理 ------------ */
    free(p->ground);  // Ground owned by host caller / ground 由主机调用方拥有，此处释放
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_outflow_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_outflow_proportion));
    CUDA_CHECK_FUNCTION(cudaFree(d_cell_height));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spill_bits));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_water_bits));
    CUDA_CHECK_FUNCTION(cudaFreeHost(h_max_spill_pinned));

    // Final sync so any pending ops complete before return.
    // 最后再 sync 一次，确保返回前所有挂起的操作都已完成。
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
