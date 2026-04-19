/*
 * Simulation of rainwater flooding
 * CUDA version - optimized
 *
 * Changes vs the baseline parallel version:
 *   (1) Rainfall: one kernel launch per cloud, grid sized to the cloud's
 *       bounding box -> threads never idle-iterate over other clouds.
 *   (2) Spillage: gather-based. Compute kernel writes per-cell
 *       outflow_level + outflow_proportion + cell_height. Apply kernel
 *       reads its own outflow and gathers inflow from 4 neighbours.
 *       Eliminates the rows*columns*4 scatter scratch buffer.
 *   (3) Per-minute max_spillage readback uses pinned memory + async copy,
 *       and the apply kernel is queued before the copy so it overlaps.
 *       Final totals (total_water, max_water) are reduced on the GPU so
 *       we don't need to copy the whole water_level matrix back.
 */

/* ==================================================================
 * HEADER INCLUDES
 * ================================================================== */
#include <float.h>      // For DBL_MAX (used as initial sentinel for max_spillage_iter)
#include <math.h>       // For mathematical functions: sqrtf, cosf, sinf, ceilf, fmaxf, M_PI
#include <stdio.h>      // For fprintf, stderr (error reporting)
#include <stdlib.h>     // For malloc, free, exit, EXIT_FAILURE (host memory management)
#include <string.h>     // For memcpy (bit-level float/int reinterpretation)

#include <cuda.h>       // CUDA runtime API: cudaMalloc, cudaMemcpy, kernel launch syntax, etc.

/* ==================================================================
 * ERROR-CHECKING MACROS
 * ==================================================================
 * These macros wrap CUDA API calls and kernel launches to check for
 * errors without cluttering the main logic. On error, they print the
 * offending line and the human-readable CUDA error string.
 * ================================================================== */

/* CUDA_CHECK_FUNCTION wraps an API call that returns cudaError_t.
 * Example: CUDA_CHECK_FUNCTION(cudaMalloc(&p, n));
 * We capture the return value into 'check' and print if non-success. */
#define CUDA_CHECK_FUNCTION(call)                                                                 \
    {                                                                                             \
        cudaError_t check = call;                                                                 \
        if (check != cudaSuccess)                                                                 \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

/* CUDA_CHECK_KERNEL polls the last error via cudaGetLastError(). Kernel
 * launches are asynchronous and use the triple-chevron syntax, so errors
 * appear as "sticky" state rather than as a direct return value. */
#define CUDA_CHECK_KERNEL()                                                                              \
    {                                                                                                    \
        cudaError_t check = cudaGetLastError();                                                          \
        if (check != cudaSuccess)                                                                        \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

#include "rng.c"        // Random number generator helpers (provided by the framework)
#include "flood.h"      // Problem parameters, results struct, macros (FIXED, FLOATING, accessMat...)

/* get_time() is a C function (defined in a .c file) used for wall-clock
 * timing. extern "C" prevents C++ name mangling so the linker can find it. */
extern "C" double get_time();

/* ==================================================================
 * THREAD-BLOCK GEOMETRY
 * ==================================================================
 * We use 16x16 = 256 threads per block. This is a sweet spot for most
 * GPUs: it matches a whole warp count (8 warps), fits nicely in
 * shared memory and registers, and gives good 2D tiling for the grid. */
#define BLOCK_X 16                         // Threads per block along the X (column) dimension
#define BLOCK_Y 16                         // Threads per block along the Y (row) dimension
#define BLOCK_THREADS (BLOCK_X * BLOCK_Y)  // Total threads per block = 256

/* ==================================================================
 * BLOCK-LEVEL REDUCTIONS
 * ==================================================================
 * Each kernel produces per-thread contributions (rain amount, water
 * loss, max spillage, etc.). Summing/maxing them at the block level
 * first reduces the number of expensive atomicAdd/atomicMax operations
 * on global memory from "one per thread" to "one per block". */

/* Sum reduction of unsigned long long across all threads of a block.
 * Result is atomically added to *global_sum (a single-location device
 * counter). Uses shared memory as a per-block scratch buffer. */
__device__ static inline void block_reduce_add_ull(
    unsigned long long value, unsigned long long *global_sum)
{
    // One slot of shared memory per thread in the block.
    __shared__ unsigned long long shared_data[BLOCK_THREADS];

    // Flatten the 2D thread index into a 1D id in [0, BLOCK_THREADS).
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread writes its own contribution.
    shared_data[threadId] = value;

    // All writes must be visible to all threads before reduction starts.
    __syncthreads();

    // Classic tree-based reduction: halve 'stride' each round, top half
    // accumulates values from bottom half until only shared_data[0] holds
    // the total for the block.
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
            shared_data[threadId] += shared_data[threadId + stride];
        __syncthreads();  // Barrier between levels of the tree.
    }

    // Only thread 0 of the block contributes to the global atomic.
    // Skip the atomic when the block-wide total is zero (common case)
    // to avoid unnecessary contention on the global counter.
    if (threadId == 0 && shared_data[0] != 0ULL)
        atomicAdd(global_sum, shared_data[0]);
}

/* Max reduction for a non-negative float across the block, written to
 * *global_max as raw int bits (atomicMax only exists for ints, so we use
 * __float_as_int() to reinterpret the bit pattern of a float).
 *
 * This trick works because for non-negative IEEE-754 floats, comparing
 * their raw bit patterns as signed integers gives the same ordering as
 * comparing them as floats. */
__device__ static inline void block_reduce_max_float_nooneg(
    float value, int *global_max)
{
    __shared__ float shared_data[BLOCK_THREADS];
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    shared_data[threadId] = value;
    __syncthreads();

    // Tree-based max reduction.
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

    // Only thread 0 does the atomic, and only if the max is > 0 (skipping
    // the zero-case is a worthwhile optimization since many blocks will
    // have no rain/spillage activity in a given minute).
    if (threadId == 0 && shared_data[0] > 0.0f)
        atomicMax(global_max, __float_as_int(shared_data[0]));
}

/* ==================================================================
 * CUDA KERNELS
 * ================================================================== */

/* ------------------------------------------------------------------
 * kernel_rain_cloud
 * ------------------------------------------------------------------
 * Rainfall from a single cloud. The CPU launches one kernel per cloud,
 * sizing the grid to the cloud's bounding box. This way threads never
 * waste time testing cells outside any cloud's radius, and no thread
 * iterates over a list of clouds.
 *
 * Parameters:
 *   water_level      - 2D grid of per-cell water amount (fixed-point int)
 *   cloud            - current cloud position/intensity/radius (by value)
 *   row_start,col_start - floating-point top-left of the bounding box in
 *                          matrix coordinates (sub-cell precision)
 *   row_first,col_first - integer top-left of the bounding box
 *   K_rows,K_cols    - bounding box dimensions in cells
 *   ex_factor        - global rainfall scaling factor
 *   rows,columns     - full grid dimensions (used by accessMat macro)
 *   d_total_rain     - global accumulator for rainfall (fixed point)
 * ------------------------------------------------------------------ */
__global__ void kernel_rain_cloud(
    int *__restrict__ water_level,    // __restrict__ = compiler hint: no aliasing
    Cloud_t cloud,                     // Pass by value -> lives in constant/param memory
    float row_start, float col_start,
    int row_first, int col_first,
    int K_rows, int K_cols,
    float ex_factor,
    int rows, int columns,
    unsigned long long *d_total_rain)
{
    // Thread's (row, col) *within* the bounding box.
    int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Keeps this thread's contribution to total rain. Stays zero if the
    // cell is outside the cloud's radius or outside the bounding box.
    int added_fixed = 0;

    // Bounding-box guard: the grid may be slightly larger than the cloud
    // box due to block-size rounding up. Threads outside the box do
    // nothing but still participate in the block reduction (with 0).
    if (local_row < K_rows && local_col < K_cols)
    {
        // Convert local box index -> absolute matrix index.
        int row_pos = row_first + local_row;
        int col_pos = col_first + local_col;

        // Sub-cell-precision matrix coordinates (as floats).
        float row_pos_f = row_start + (float)local_row;
        float col_pos_f = col_start + (float)local_col;

        // Convert matrix coords -> scenario (km) coords.
        float x_pos = COORD_MAT2SCEN_X(col_pos_f);
        float y_pos = COORD_MAT2SCEN_Y(row_pos_f);

        // Distance from this cell to the cloud's centre (in km).
        float dx = x_pos - cloud.x;
        float dy = y_pos - cloud.y;
        float distance = sqrtf(dx * dx + dy * dy);

        // Only cells inside the cloud's radius get rain.
        if (distance < cloud.radius)
        {
            // Rain intensity falls off linearly with normalised distance.
            // fmaxf(0, ...) guards against underflow near the edge.
            float rain = ex_factor *
                         fmaxf(0.0f, cloud.intensity - distance / cloud.radius * sqrtf(cloud.intensity));

            // Convert mm/hour -> meters/minute for accumulation.
            float meters_per_minute = rain / 1000.0f / 60.0f;

            // Convert float (meters) -> fixed-point int (micro-meters).
            added_fixed = FIXED(meters_per_minute);

            // Write the rain into the global water_level matrix.
            // Skip the write if the increment is exactly zero (tiny save).
            if (added_fixed != 0)
                accessMat(water_level, row_pos, col_pos) += added_fixed;
        }
    }

    // Every thread (even those outside the box) joins the block reduction.
    // This is required because __syncthreads inside the reduction must be
    // reached by all threads of the block.
    block_reduce_add_ull((unsigned long long)added_fixed, d_total_rain);
}

/* ------------------------------------------------------------------
 * kernel_spillage_compute
 * ------------------------------------------------------------------
 * First half of the spillage step. For each cell it computes:
 *   outflow_level       - amount of water that this cell will shed
 *   outflow_proportion  - the fraction proportional factor used later
 *   cell_height         - ground + water (cached so the apply kernel
 *                          doesn't need to re-read water_level, which
 *                          is being modified in place)
 *
 * It also accumulates:
 *   d_total_loss    - fixed-point total water lost off the border
 *   d_max_spill_bits - global max outflow_level (int-bits encoded)
 *
 * The split into two kernels is what makes the spillage race-free. In
 * the baseline scatter implementation, apply would write directly into
 * neighbour water_level cells (data race). Here each cell only updates
 * its own water_level (gather), using neighbour state that was frozen
 * by compute, so no atomic writes are needed.
 * ------------------------------------------------------------------ */
__global__ void kernel_spillage_compute(
    const int *__restrict__ water_level,       // READ-ONLY in this kernel
    const float *__restrict__ ground,           // Terrain heights (static)
    float *__restrict__ outflow_level,          // OUTPUT: per-cell outflow
    float *__restrict__ outflow_proportion,     // OUTPUT: redistribution factor
    float *__restrict__ cell_height,            // OUTPUT: ground + water (cached)
    unsigned long long *d_total_loss,           // Reduction: total boundary loss
    int *d_max_spill_bits,                      // Reduction: max outflow
    int rows, int columns)
{
    // Absolute grid coordinates for this thread.
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // Per-thread scratch values. Defaults used for out-of-range threads.
    long local_loss = 0;                // This cell's border-loss contribution
    float local_spillage_level = 0.0f;  // This cell's outflow amount
    float computed_proportion = 0.0f;   // This cell's redistribution factor

    // In-bounds guard.
    if (row_pos < rows && col_pos < columns)
    {
        // Read this cell's water (fixed-point int) and ground (float).
        int w = accessMat(water_level, row_pos, col_pos);
        float ground_self = accessMat(ground, row_pos, col_pos);
        float water_f = FLOATING(w);                  // int -> float in meters
        float current_height = ground_self + water_f; // Total surface height

        // Only cells with water may spill. Dry cells contribute 0.
        if (w > 0)
        {
            float sum_diff = 0.0f;  // Total height difference to all "lower" neighbours.

            // Loop over the 4 neighbours (up, down, left, right).
            // #pragma unroll hints the compiler to fully unroll this loop
            // of compile-time-known length for better scheduling.
#pragma unroll
            for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
            {
                int nr = row_pos + displacements[cp][0];
                int nc = col_pos + displacements[cp][1];
                float nh;
                // Boundary handling: treat out-of-grid neighbours as if
                // they had the same ground height but zero water. That
                // way, water at the edge "falls off" the grid.
                if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
                    nh = ground_self;
                else
                    nh = accessMat(ground, nr, nc) +
                         FLOATING(accessMat(water_level, nr, nc));

                // Only neighbours at or below our height receive spill.
                if (current_height >= nh)
                {
                    float hd = current_height - nh;
                    sum_diff += hd;
                    // The maximum height-drop becomes our spillage level
                    // (capped below by available water).
                    if (hd > local_spillage_level)
                        local_spillage_level = hd;
                }
            }

            // Can't spill more water than we have.
            if (water_f < local_spillage_level)
                local_spillage_level = water_f;

            // If we have any downhill neighbour, compute the proportion.
            if (sum_diff > 0.0f)
            {
                float prop = local_spillage_level / sum_diff;
                // Tiny-proportion filter: avoids denormal mayhem and
                // ensures the apply step's tests are meaningful.
                if (prop > 1e-8f)
                {
                    computed_proportion = prop;
                    // Count water lost through grid boundaries. Each
                    // out-of-grid "neighbour" contributes half of the
                    // head difference times the proportion.
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
                    // Proportion too small: pretend we spill nothing.
                    local_spillage_level = 0.0f;
                }
            }
            else
            {
                // No downhill neighbour: nothing to spill.
                local_spillage_level = 0.0f;
            }
        }

        // Write the three outputs for this cell.
        accessMat(outflow_level, row_pos, col_pos) = local_spillage_level;
        accessMat(outflow_proportion, row_pos, col_pos) = computed_proportion;
        accessMat(cell_height, row_pos, col_pos) = current_height;
    }

    // Block-level reductions into the two global counters. All threads
    // in the block must reach these calls for correctness.
    block_reduce_add_ull((unsigned long long)local_loss, d_total_loss);
    block_reduce_max_float_nooneg(local_spillage_level, d_max_spill_bits);
}

/* ------------------------------------------------------------------
 * kernel_spillage_apply
 * ------------------------------------------------------------------
 * Second half of the spillage step. GATHER-based:
 *   delta = -own_outflow/SPILLAGE_FACTOR
 *         + sum over neighbours of (n_prop * (n_height - my_height))/SPILLAGE_FACTOR
 * Each cell writes ONLY to its own water_level[i] location. That means
 * no atomicAdd and no race even though every thread reads its 4
 * neighbours' state, because those reads go through the static
 * outflow_proportion and cell_height buffers that compute produced.
 * ------------------------------------------------------------------ */
__global__ void kernel_spillage_apply(
    int *__restrict__ water_level,               // WRITE: updated water
    const float *__restrict__ cell_height,        // Frozen total heights
    const float *__restrict__ outflow_level,      // Own outflow amount
    const float *__restrict__ outflow_proportion, // Neighbour inflow factor
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    // Early-out: threads outside the grid have no work and (unlike in
    // compute) don't participate in any reduction here, so we can just
    // return.
    if (row_pos >= rows || col_pos >= columns)
        return;

    // Pull our own height and outflow from the cached buffers.
    float my_height = accessMat(cell_height, row_pos, col_pos);
    float my_outflow = accessMat(outflow_level, row_pos, col_pos);

    // Net change in fixed-point water units. Starts at zero.
    int delta = 0;

    // Step 1: subtract our own outflow (scaled by SPILLAGE_FACTOR).
    if (my_outflow > 0.0f)
        delta -= FIXED(my_outflow / SPILLAGE_FACTOR);

    // Step 2: add the inflow we receive from each of the 4 neighbours.
#pragma unroll
    for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
    {
        int nr = row_pos + displacements[cp][0];
        int nc = col_pos + displacements[cp][1];
        float infl = 0.0f;
        // In-bounds neighbours only.
        if (nr >= 0 && nr < rows && nc >= 0 && nc < columns)
        {
            float n_prop = accessMat(outflow_proportion, nr, nc);
            if (n_prop > 0.0f)
            {
                float nh = accessMat(cell_height, nr, nc);
                // Only receive from neighbours that are higher than us
                // (water flows downhill).
                if (nh >= my_height)
                    infl = n_prop * (nh - my_height);
            }
        }
        delta += FIXED(infl / SPILLAGE_FACTOR);
    }

    // Single write per cell. No atomic needed because no other thread
    // writes to water_level[row_pos, col_pos].
    if (delta != 0)
        accessMat(water_level, row_pos, col_pos) += delta;
}

/* ------------------------------------------------------------------
 * kernel_final_stats
 * ------------------------------------------------------------------
 * Runs once, after the main loop, to compute total_water and
 * max_water_scenario on the GPU. This saves copying the whole
 * water_level matrix back to the host just to run a simple reduction.
 *
 * Implements sum+max in a single pass using two shared-memory arrays.
 * ------------------------------------------------------------------ */
__global__ void kernel_final_stats(
    const int *__restrict__ water_level,
    unsigned long long *d_total_water,   // Sum of all fixed-point waters
    int *d_max_water_bits,                // Max water (float bits)
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    long long local_water = 0;   // Per-thread sum contribution
    float local_max_f = 0.0f;    // Per-thread max contribution
    if (row_pos < rows && col_pos < columns)
    {
        int w = accessMat(water_level, row_pos, col_pos);
        local_water = w;              // Accumulate raw fixed-point units
        local_max_f = FLOATING(w);    // Convert to float for the max
    }

    // Two shared-memory buffers -> one tree reduction for sum AND max.
    __shared__ long long sh_sum[BLOCK_THREADS];
    __shared__ float sh_max[BLOCK_THREADS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sh_sum[tid] = local_water;
    sh_max[tid] = local_max_f;
    __syncthreads();

    // Combined tree reduction: every iteration halves the active set.
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

    // Thread 0 of every block contributes to the global atomics.
    if (tid == 0)
    {
        atomicAdd(d_total_water, (unsigned long long)sh_sum[0]);
        if (sh_max[0] > 0.0f)
            atomicMax(d_max_water_bits, __float_as_int(sh_max[0]));
    }
}

/* ==================================================================
 * HOST-SIDE DRIVER: do_compute
 * ==================================================================
 * This is the function called by the framework. It:
 *   1. Allocates device memory
 *   2. Runs the main simulation loop (rain + spillage) for up to
 *      num_minutes iterations, or until max_spillage drops below
 *      the configured threshold.
 *   3. Produces final statistics.
 *   4. Frees everything.
 * ================================================================== */
extern "C" void
do_compute(struct parameters *p, struct results *r)
{
    // Cache the grid dimensions and pointer to the iteration counter.
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    // Pick GPU 0 and ensure it's idle before we start measuring.
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    // Total number of cells. size_t because rows*columns may exceed int.
    size_t n_cells = (size_t)rows * (size_t)columns;

    /* ------------ Device pointers ------------ */
    int *d_water_level;                                                    // Per-cell fixed-point water
    float *d_ground, *d_outflow_level, *d_outflow_proportion, *d_cell_height; // 4 float grids
    unsigned long long *d_total_rain, *d_total_loss, *d_total_water;       // Global counters
    int *d_max_spill_bits, *d_max_water_bits;                              // Float-bits max counters

    /* ------------ Allocate device memory ------------ */
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

    /* ------------ Zero-initialise all grids and counters ------------
     * cudaMemset is a fast device-side bulk zeroing. Note: we do NOT
     * zero d_max_spill_bits here because it's reset at the top of each
     * minute's spillage step. */
    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, sizeof(int) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_level, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_proportion, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_cell_height, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_rain, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_loss, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_water, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_max_water_bits, 0, sizeof(int)));

    // Copy the static ground heights from host to device (one-shot).
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, p->ground, sizeof(float) * n_cells, cudaMemcpyHostToDevice));

    /* ------------ Pinned host buffer for max_spill readback ------------
     * Pinned (page-locked) memory enables true async DMA transfers and
     * avoids the runtime staging step that non-pinned memory requires.
     * It lets the memcpy overlap with kernel execution. */
    int *h_max_spill_pinned;
    CUDA_CHECK_FUNCTION(cudaHostAlloc(&h_max_spill_pinned, sizeof(int), cudaHostAllocDefault));

    // Loop-termination sentinel: start with "infinity" so the first
    // iteration always runs.
    double max_spillage_iter = DBL_MAX;

    // Thread-block shape and full-grid size (covers the whole matrix).
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_full((columns + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Start the wall-clock timer.
    r->runtime = get_time();

    /* ==============================================================
     * MAIN SIMULATION LOOP
     * Runs until either num_minutes is reached, or the last minute's
     * max spillage drops below the convergence threshold.
     * ============================================================== */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++)
    {
        /* Step 1: move clouds (CPU, small work).
         * num_clouds is typically small (tens to hundreds), so we
         * update them on the host. This avoids kernel-launch overhead
         * for a negligible amount of work. */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            Cloud_t *c = &p->clouds[cloud];
            // Convert km/h speed to km/minute.
            float km_minute = c->speed / 60.0f;
            // Advance along the cloud's angle.
            c->x += km_minute * cosf(c->angle * (float)M_PI / 180.0f);
            c->y += km_minute * sinf(c->angle * (float)M_PI / 180.0f);
        }

        /* Step 2: rainfall, one kernel per cloud over its bounding box.
         * Each cloud is independent - kernels go into the default
         * stream and execute in order, but threads within each kernel
         * are naturally parallel. */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            // Copy the cloud struct (pass by value into the kernel).
            Cloud_t c = p->clouds[cloud];

            // Compute the cloud's axis-aligned bounding box in matrix
            // coordinates, clipped to the scenario. Float values allow
            // sub-cell precision when the cloud doesn't align to grid.
            float row_start = COORD_SCEN2MAT_Y(MAX(0.0f, c.y - c.radius));
            float row_end   = COORD_SCEN2MAT_Y(MIN(c.y + c.radius, (float)SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0.0f, c.x - c.radius));
            float col_end   = COORD_SCEN2MAT_X(MIN(c.x + c.radius, (float)SCENARIO_SIZE));

            // Integer top-left cell, and cell-count extents. ceilf is
            // safer than a truncating cast so we don't miss a partial row.
            int row_first = (int)row_start;
            int col_first = (int)col_start;
            int K_rows = (int)ceilf(row_end - row_start);
            int K_cols = (int)ceilf(col_end - col_start);

            // If the cloud is entirely off-grid, skip it.
            if (K_rows <= 0 || K_cols <= 0)
                continue;

            // Clamp start indices to 0 (clouds can wander off-grid).
            if (row_first < 0)
                row_first = 0;
            if (col_first < 0)
                col_first = 0;
            // Clamp extents to the grid's actual size.
            if (row_first + K_rows > rows)
                K_rows = rows - row_first;
            if (col_first + K_cols > columns)
                K_cols = columns - col_first;
            if (K_rows <= 0 || K_cols <= 0)
                continue;

            // Grid dimensions sized exactly to the clamped bounding box.
            dim3 grid_cloud((K_cols + block.x - 1) / block.x,
                            (K_rows + block.y - 1) / block.y);

            // Launch the rainfall kernel for this one cloud.
            kernel_rain_cloud<<<grid_cloud, block>>>(
                d_water_level, c, row_start, col_start,
                row_first, col_first, K_rows, K_cols,
                p->ex_factor, rows, columns, d_total_rain);
        }

        /* Step 3: spillage compute.
         * Reset max_spill to 0 before the compute kernel (async, so it
         * overlaps with previous work queued in the stream). */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spill_bits, 0, sizeof(int)));
        kernel_spillage_compute<<<grid_full, block>>>(
            d_water_level, d_ground,
            d_outflow_level, d_outflow_proportion, d_cell_height,
            d_total_loss, d_max_spill_bits, rows, columns);

        /* Step 4: spillage apply. Queued before the max_spill readback
         * so the copy can overlap with kernel execution. */
        kernel_spillage_apply<<<grid_full, block>>>(
            d_water_level, d_cell_height,
            d_outflow_level, d_outflow_proportion, rows, columns);

        /* Step 5: async readback + synchronous wait (pinned host memory).
         * cudaMemcpyAsync + pinned host memory = DMA-driven copy that
         * overlaps with the apply kernel. We then sync to be sure the
         * value is ready before we inspect it on the host. */
        CUDA_CHECK_FUNCTION(cudaMemcpyAsync(
            h_max_spill_pinned, d_max_spill_bits, sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK_FUNCTION(cudaStreamSynchronize(0));

        // Decode the max_spill value. We stored it as int-bits of a
        // float; zero is a legitimate "no spillage" marker.
        float max_sl;
        if (*h_max_spill_pinned == 0)
            max_sl = 0.0f;
        else
            memcpy(&max_sl, h_max_spill_pinned, sizeof(int));

        // Divide by SPILLAGE_FACTOR to get the per-minute maximum.
        max_spillage_iter = (double)max_sl / SPILLAGE_FACTOR;

        // Track the overall-max across the simulation.
        if (max_spillage_iter > r->max_spillage_scenario)
        {
            r->max_spillage_scenario = max_spillage_iter;
            r->max_spillage_minute = *minute;
        }
    }

    /* Final reduction on the GPU: compute total_water and max_water_scenario
     * without ever copying the full water_level matrix back to the host. */
    kernel_final_stats<<<grid_full, block>>>(
        d_water_level, d_total_water, d_max_water_bits, rows, columns);

    // Pull the four scalar reductions back to the host.
    unsigned long long h_total_rain = 0, h_total_loss = 0, h_total_water = 0;
    int h_max_water_bits = 0;
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_rain, d_total_rain, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_loss, d_total_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_water, d_total_water, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_max_water_bits, d_max_water_bits, sizeof(int), cudaMemcpyDeviceToHost));

    // Ensure everything is finished, then stop the timer.
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    r->runtime = get_time() - r->runtime;

    // Populate the results struct. Note the casts: we stored as unsigned
    // but the framework uses signed long.
    r->total_rain += (long)h_total_rain;
    r->total_water_loss += (long)h_total_loss;
    r->total_water = (long)h_total_water;

    // Decode the max_water_scenario (int-bits -> float).
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

    /* Optional: dump the final water matrix if requested by command line.
     * This is a developer/test feature; in production runs final_matrix
     * is 0 and we skip the big host copy + print. */
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

    /* ------------ Cleanup ------------ */
    free(p->ground);  // Ground was owned by the host caller, free it now.
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

    // Final synchronise so any pending operations complete before return.
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
