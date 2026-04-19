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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

#define CUDA_CHECK_FUNCTION(call)                                                                 \
    {                                                                                             \
        cudaError_t check = call;                                                                 \
        if (check != cudaSuccess)                                                                 \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }
#define CUDA_CHECK_KERNEL()                                                                              \
    {                                                                                                    \
        cudaError_t check = cudaGetLastError();                                                          \
        if (check != cudaSuccess)                                                                        \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check)); \
    }

#include "rng.c"
#include "flood.h"

extern "C" double get_time();

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_THREADS (BLOCK_X * BLOCK_Y)

/* -------- block reductions -------- */
__device__ static inline void block_reduce_add_ull(
    unsigned long long value, unsigned long long *global_sum)
{
    __shared__ unsigned long long shared_data[BLOCK_THREADS];
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    shared_data[threadId] = value;
    __syncthreads();
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
            shared_data[threadId] += shared_data[threadId + stride];
        __syncthreads();
    }
    if (threadId == 0 && shared_data[0] != 0ULL)
        atomicAdd(global_sum, shared_data[0]);
}

__device__ static inline void block_reduce_max_float_nooneg(
    float value, int *global_max)
{
    __shared__ float shared_data[BLOCK_THREADS];
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    shared_data[threadId] = value;
    __syncthreads();
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
    if (threadId == 0 && shared_data[0] > 0.0f)
        atomicMax(global_max, __float_as_int(shared_data[0]));
}

/* -------- kernels -------- */

/* Rainfall for a single cloud. Grid only covers the cloud bounding box. */
__global__ void kernel_rain_cloud(
    int *__restrict__ water_level,
    Cloud_t cloud,
    float row_start, float col_start,
    int row_first, int col_first,
    int K_rows, int K_cols,
    float ex_factor,
    int rows, int columns,
    unsigned long long *d_total_rain)
{
    int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    int added_fixed = 0;

    if (local_row < K_rows && local_col < K_cols)
    {
        int row_pos = row_first + local_row;
        int col_pos = col_first + local_col;

        float row_pos_f = row_start + (float)local_row;
        float col_pos_f = col_start + (float)local_col;
        float x_pos = COORD_MAT2SCEN_X(col_pos_f);
        float y_pos = COORD_MAT2SCEN_Y(row_pos_f);
        float dx = x_pos - cloud.x;
        float dy = y_pos - cloud.y;
        float distance = sqrtf(dx * dx + dy * dy);
        if (distance < cloud.radius)
        {
            float rain = ex_factor *
                         fmaxf(0.0f, cloud.intensity - distance / cloud.radius * sqrtf(cloud.intensity));
            float meters_per_minute = rain / 1000.0f / 60.0f;
            added_fixed = FIXED(meters_per_minute);
            if (added_fixed != 0)
                accessMat(water_level, row_pos, col_pos) += added_fixed;
        }
    }
    block_reduce_add_ull((unsigned long long)added_fixed, d_total_rain);
}

/* Spillage compute: writes outflow_level, outflow_proportion, cell_height
 * for every cell, plus reduces total_loss (border losses) and max_spill. */
__global__ void kernel_spillage_compute(
    const int *__restrict__ water_level,
    const float *__restrict__ ground,
    float *__restrict__ outflow_level,
    float *__restrict__ outflow_proportion,
    float *__restrict__ cell_height,
    unsigned long long *d_total_loss,
    int *d_max_spill_bits,
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    long local_loss = 0;
    float local_spillage_level = 0.0f;
    float computed_proportion = 0.0f;

    if (row_pos < rows && col_pos < columns)
    {
        int w = accessMat(water_level, row_pos, col_pos);
        float ground_self = accessMat(ground, row_pos, col_pos);
        float water_f = FLOATING(w);
        float current_height = ground_self + water_f;

        if (w > 0)
        {
            float sum_diff = 0.0f;

#pragma unroll
            for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
            {
                int nr = row_pos + displacements[cp][0];
                int nc = col_pos + displacements[cp][1];
                float nh;
                if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
                    nh = ground_self;
                else
                    nh = accessMat(ground, nr, nc) +
                         FLOATING(accessMat(water_level, nr, nc));

                if (current_height >= nh)
                {
                    float hd = current_height - nh;
                    sum_diff += hd;
                    if (hd > local_spillage_level)
                        local_spillage_level = hd;
                }
            }
            if (water_f < local_spillage_level)
                local_spillage_level = water_f;

            if (sum_diff > 0.0f)
            {
                float prop = local_spillage_level / sum_diff;
                if (prop > 1e-8f)
                {
                    computed_proportion = prop;
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
                    local_spillage_level = 0.0f;
                }
            }
            else
            {
                local_spillage_level = 0.0f;
            }
        }

        accessMat(outflow_level, row_pos, col_pos) = local_spillage_level;
        accessMat(outflow_proportion, row_pos, col_pos) = computed_proportion;
        accessMat(cell_height, row_pos, col_pos) = current_height;
    }

    block_reduce_add_ull((unsigned long long)local_loss, d_total_loss);
    block_reduce_max_float_nooneg(local_spillage_level, d_max_spill_bits);
}

/* Spillage apply: gather based. Reads own outflow and 4 neighbours'
 * (proportion, height). Writes water_level in place.
 * Race-free because neighbour heights come from the cached cell_height
 * buffer, not from water_level. */
__global__ void kernel_spillage_apply(
    int *__restrict__ water_level,
    const float *__restrict__ cell_height,
    const float *__restrict__ outflow_level,
    const float *__restrict__ outflow_proportion,
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_pos >= rows || col_pos >= columns)
        return;

    float my_height = accessMat(cell_height, row_pos, col_pos);
    float my_outflow = accessMat(outflow_level, row_pos, col_pos);

    int delta = 0;
    if (my_outflow > 0.0f)
        delta -= FIXED(my_outflow / SPILLAGE_FACTOR);

#pragma unroll
    for (int cp = 0; cp < CONTIGUOUS_CELLS; cp++)
    {
        int nr = row_pos + displacements[cp][0];
        int nc = col_pos + displacements[cp][1];
        float infl = 0.0f;
        if (nr >= 0 && nr < rows && nc >= 0 && nc < columns)
        {
            float n_prop = accessMat(outflow_proportion, nr, nc);
            if (n_prop > 0.0f)
            {
                float nh = accessMat(cell_height, nr, nc);
                if (nh >= my_height)
                    infl = n_prop * (nh - my_height);
            }
        }
        delta += FIXED(infl / SPILLAGE_FACTOR);
    }

    if (delta != 0)
        accessMat(water_level, row_pos, col_pos) += delta;
}

/* Final-pass reduction: total_water and max_water_scenario on the GPU. */
__global__ void kernel_final_stats(
    const int *__restrict__ water_level,
    unsigned long long *d_total_water,
    int *d_max_water_bits,
    int rows, int columns)
{
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;

    long long local_water = 0;
    float local_max_f = 0.0f;
    if (row_pos < rows && col_pos < columns)
    {
        int w = accessMat(water_level, row_pos, col_pos);
        local_water = w;
        local_max_f = FLOATING(w);
    }

    __shared__ long long sh_sum[BLOCK_THREADS];
    __shared__ float sh_max[BLOCK_THREADS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sh_sum[tid] = local_water;
    sh_max[tid] = local_max_f;
    __syncthreads();
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
    if (tid == 0)
    {
        atomicAdd(d_total_water, (unsigned long long)sh_sum[0]);
        if (sh_max[0] > 0.0f)
            atomicMax(d_max_water_bits, __float_as_int(sh_max[0]));
    }
}

/* -------- host-side do_compute -------- */
extern "C" void
do_compute(struct parameters *p, struct results *r)
{
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    size_t n_cells = (size_t)rows * (size_t)columns;

    int *d_water_level;
    float *d_ground, *d_outflow_level, *d_outflow_proportion, *d_cell_height;
    unsigned long long *d_total_rain, *d_total_loss, *d_total_water;
    int *d_max_spill_bits, *d_max_water_bits;

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

    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, sizeof(int) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_level, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_outflow_proportion, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_cell_height, 0, sizeof(float) * n_cells));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_rain, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_loss, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_water, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_max_water_bits, 0, sizeof(int)));

    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, p->ground, sizeof(float) * n_cells, cudaMemcpyHostToDevice));

    /* Pinned host buffer for async max_spill readback. */
    int *h_max_spill_pinned;
    CUDA_CHECK_FUNCTION(cudaHostAlloc(&h_max_spill_pinned, sizeof(int), cudaHostAllocDefault));

    double max_spillage_iter = DBL_MAX;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_full((columns + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    r->runtime = get_time();

    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++)
    {
        /* Step 1: move clouds (CPU, small work). */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            Cloud_t *c = &p->clouds[cloud];
            float km_minute = c->speed / 60.0f;
            c->x += km_minute * cosf(c->angle * (float)M_PI / 180.0f);
            c->y += km_minute * sinf(c->angle * (float)M_PI / 180.0f);
        }

        /* Step 2: rainfall, one kernel per cloud over its bounding box. */
        for (int cloud = 0; cloud < p->num_clouds; cloud++)
        {
            Cloud_t c = p->clouds[cloud];
            float row_start = COORD_SCEN2MAT_Y(MAX(0.0f, c.y - c.radius));
            float row_end = COORD_SCEN2MAT_Y(MIN(c.y + c.radius, (float)SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0.0f, c.x - c.radius));
            float col_end = COORD_SCEN2MAT_X(MIN(c.x + c.radius, (float)SCENARIO_SIZE));
            int row_first = (int)row_start;
            int col_first = (int)col_start;
            int K_rows = (int)ceilf(row_end - row_start);
            int K_cols = (int)ceilf(col_end - col_start);
            if (K_rows <= 0 || K_cols <= 0)
                continue;
            /* Clamp to grid bounds. */
            if (row_first < 0)
                row_first = 0;
            if (col_first < 0)
                col_first = 0;
            if (row_first + K_rows > rows)
                K_rows = rows - row_first;
            if (col_first + K_cols > columns)
                K_cols = columns - col_first;
            if (K_rows <= 0 || K_cols <= 0)
                continue;

            dim3 grid_cloud((K_cols + block.x - 1) / block.x,
                            (K_rows + block.y - 1) / block.y);
            kernel_rain_cloud<<<grid_cloud, block>>>(
                d_water_level, c, row_start, col_start,
                row_first, col_first, K_rows, K_cols,
                p->ex_factor, rows, columns, d_total_rain);
        }

        /* Step 3: spillage compute. */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spill_bits, 0, sizeof(int)));
        kernel_spillage_compute<<<grid_full, block>>>(
            d_water_level, d_ground,
            d_outflow_level, d_outflow_proportion, d_cell_height,
            d_total_loss, d_max_spill_bits, rows, columns);

        /* Step 4: spillage apply. Queued before the max_spill readback so
         * the copy can overlap with kernel execution. */
        kernel_spillage_apply<<<grid_full, block>>>(
            d_water_level, d_cell_height,
            d_outflow_level, d_outflow_proportion, rows, columns);

        /* Step 5: async readback + synchronous wait (pinned host memory). */
        CUDA_CHECK_FUNCTION(cudaMemcpyAsync(
            h_max_spill_pinned, d_max_spill_bits, sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK_FUNCTION(cudaStreamSynchronize(0));

        float max_sl;
        if (*h_max_spill_pinned == 0)
            max_sl = 0.0f;
        else
            memcpy(&max_sl, h_max_spill_pinned, sizeof(int));

        max_spillage_iter = (double)max_sl / SPILLAGE_FACTOR;

        if (max_spillage_iter > r->max_spillage_scenario)
        {
            r->max_spillage_scenario = max_spillage_iter;
            r->max_spillage_minute = *minute;
        }
    }

    /* Final reduction on the GPU. */
    kernel_final_stats<<<grid_full, block>>>(
        d_water_level, d_total_water, d_max_water_bits, rows, columns);

    unsigned long long h_total_rain = 0, h_total_loss = 0, h_total_water = 0;
    int h_max_water_bits = 0;
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_rain, d_total_rain, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_loss, d_total_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_water, d_total_water, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&h_max_water_bits, d_max_water_bits, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    r->runtime = get_time() - r->runtime;

    r->total_rain += (long)h_total_rain;
    r->total_water_loss += (long)h_total_loss;
    r->total_water = (long)h_total_water;

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

    free(p->ground);
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

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
