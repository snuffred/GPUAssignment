/*
 * NOTE: READ CAREFULLY
 * Here the function `do_compute` is just a copy of the CPU sequential version.
 * Implement your GPU code with CUDA here. Check the README for further instructions.
 * You can modify everything in this file, as long as we can compile the executable using
 * this source code, and Makefile.
 *
 * Simulation of rainwater flooding
 * CUDA version (Implement your parallel version here)
 *
 * Adapted for ACCE at the VU, Period 5 2025-2026 from the original version by
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Headers for the CUDA assignment versions */
#include <cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION(call)                                                                                      \
    {                                                                                                                  \
        cudaError_t check = call;                                                                                      \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));                      \
    }
#define CUDA_CHECK_KERNEL()                                                                                            \
    {                                                                                                                  \
        cudaError_t check = cudaGetLastError();                                                                        \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));               \
    }

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 * Header file: Contains constants and definitions
 */
#include "flood.h"

extern "C" double get_time();

/*
 * Main compute function
 */
extern "C" void do_compute(struct parameters *p, struct results *r) {
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    /* 2. Start global timer */
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    /*
     *
     * Allocate memory and call kernels in this function.
     * Ensure all debug and animation code works in your final version.
     *
     */

    /* Memory allocation */

    int *water_level;           // Level of water on each cell (fixed precision)
    float *ground;              // Ground height
    float *spillage_flag;       // Indicates which cells are spilling to neighbors
    float *spillage_level;      // Maximum level of spillage of each cell
    float *spillage_from_neigh; // Spillage from each neighbor

    ground = p->ground;
    water_level = (int *)malloc(sizeof(int) * (size_t)rows * (size_t)columns);
    spillage_flag = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)columns);
    spillage_level = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)columns);
    spillage_from_neigh = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)columns * (size_t)CONTIGUOUS_CELLS);

    if (water_level == NULL || spillage_flag == NULL || spillage_level == NULL || spillage_from_neigh == NULL) {
        fprintf(stderr, "-- Error allocating ground and rain structures for size: %d x %d \n", rows, columns);
        exit(EXIT_FAILURE);
    }

    /* Ground generation and initialization of other structures */
    int row_pos, col_pos, depth_pos;
    for (row_pos = 0; row_pos < rows; row_pos++) {
        for (col_pos = 0; col_pos < columns; col_pos++) {
            accessMat(water_level, row_pos, col_pos) = 0;
            accessMat(spillage_flag, row_pos, col_pos) = 0.0;
            accessMat(spillage_level, row_pos, col_pos) = 0.0;
            int depths = CONTIGUOUS_CELLS;
            for (depth_pos = 0; depth_pos < depths; depth_pos++)
                accessMat3D(spillage_from_neigh, row_pos, col_pos, depth_pos) = 0.0;
        }
    }

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    double max_spillage_iter = DBL_MAX;

    /* Prepare to measure runtime */
    r->runtime = get_time();

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++) {

        int new_row, new_col;
        int cell_pos;

        /* Step 1.1: Clouds movement */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            // Calculate new position (x are rows, y are columns)
            Cloud_t *c_cloud = &p->clouds[cloud];
            float km_minute = c_cloud->speed / 60;
            c_cloud->x += km_minute * cos(c_cloud->angle * M_PI / 180.0);
            c_cloud->y += km_minute * sin(c_cloud->angle * M_PI / 180.0);
        }

#ifdef DEBUG
#ifndef ANIMATION
        print_clouds(p->num_clouds, p->clouds);
#endif
#endif

        /* Step 1.2: Rainfall */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            Cloud_t c_cloud = p->clouds[cloud];
            // Compute the bounding box area of the cloud
            float row_start = COORD_SCEN2MAT_Y(MAX(0, c_cloud.y - c_cloud.radius));
            float row_end = COORD_SCEN2MAT_Y(MIN(c_cloud.y + c_cloud.radius, SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0, c_cloud.x - c_cloud.radius));
            float col_end = COORD_SCEN2MAT_X(MIN(c_cloud.x + c_cloud.radius, SCENARIO_SIZE));
            float distance;

            // Add rain to the ground water level
            float row_pos, col_pos;
            for (row_pos = row_start; row_pos < row_end; row_pos++) {
                for (col_pos = col_start; col_pos < col_end; col_pos++) {
                    float x_pos = COORD_MAT2SCEN_X(col_pos);
                    float y_pos = COORD_MAT2SCEN_Y(row_pos);
                    distance =
                        sqrt((x_pos - c_cloud.x) * (x_pos - c_cloud.x) + (y_pos - c_cloud.y) * (y_pos - c_cloud.y));
                    if (distance < c_cloud.radius) {
                        float rain = p->ex_factor *
                                     MAX(0, c_cloud.intensity - distance / c_cloud.radius * sqrt(c_cloud.intensity));
                        float meters_per_minute = rain / 1000 / 60;
                        accessMat(water_level, row_pos, col_pos) += FIXED(meters_per_minute);
                        r->total_rain += FIXED(meters_per_minute);
                    }
                }
            }
        }

#ifdef DEBUG
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

        /* Step 2: Compute water spillage to neighbor cells */
        for (row_pos = 0; row_pos < rows; row_pos++) {
            for (col_pos = 0; col_pos < columns; col_pos++) {
                if (accessMat(water_level, row_pos, col_pos) > 0) {
                    float sum_diff = 0;
                    float my_spillage_level = 0;

                    /* Differences between current-cell level and its neighbours  */
                    float current_height =
                        accessMat(ground, row_pos, col_pos) + FLOATING(accessMat(water_level, row_pos, col_pos));

                    // Iterate over the four neighboring cells using the displacement array
                    for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                        new_row = row_pos + displacements[cell_pos][0];
                        new_col = col_pos + displacements[cell_pos][1];

                        float neighbor_height;

                        // Check if the new position is within the matrix boundaries
                        if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns)
                            // Out of borders: Same height as the cell with no water
                            neighbor_height = accessMat(ground, row_pos, col_pos);
                        else
                            // Neighbor cell: Ground height + water level
                            neighbor_height = accessMat(ground, new_row, new_col) +
                                              FLOATING(accessMat(water_level, new_row, new_col));

                        // Compute level differences
                        if (current_height >= neighbor_height) {
                            float height_diff = current_height - neighbor_height;
                            sum_diff += height_diff;
                            my_spillage_level = MAX(my_spillage_level, height_diff);
                        }
                    }
                    my_spillage_level = MIN(FLOATING(accessMat(water_level, row_pos, col_pos)), my_spillage_level);

                    // Compute proportion of spillage to each neighbor
                    if (sum_diff > 0.0) {
                        float proportion = my_spillage_level / sum_diff;
                        // If proportion is significative, spillage
                        if (proportion > 1e-8) {
                            accessMat(spillage_flag, row_pos, col_pos) = 1;
                            accessMat(spillage_level, row_pos, col_pos) = my_spillage_level;

                            // Iterate over the four neighboring cells using the displacement array
                            for (cell_pos = 0; cell_pos < 4; cell_pos++) {
                                new_row = row_pos + displacements[cell_pos][0];
                                new_col = col_pos + displacements[cell_pos][1];

                                float neighbor_height;

                                // Check if the new position is within the matrix boundaries
                                if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
                                    // Spillage out of the borders: Water loss
                                    neighbor_height = accessMat(ground, row_pos, col_pos);
                                    if (current_height >= neighbor_height) {
                                        r->total_water_loss +=
                                            FIXED(proportion * (current_height - neighbor_height) / 2);
                                    }
                                } else {
                                    // Spillage to a neighbor cell
                                    neighbor_height = accessMat(ground, new_row, new_col) +
                                                      FLOATING(accessMat(water_level, new_row, new_col));
                                    if (current_height >= neighbor_height) {
                                        int depths = CONTIGUOUS_CELLS;
                                        accessMat3D(spillage_from_neigh, new_row, new_col, cell_pos) =
                                            proportion * (current_height - neighbor_height);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /* Step 3: Propagation of previuosly computer water spillage to/from neighbors */
        max_spillage_iter = 0.0;
        for (row_pos = 0; row_pos < rows; row_pos++) {
            for (col_pos = 0; col_pos < columns; col_pos++) {
                // If the cell has spillage
                if (accessMat(spillage_flag, row_pos, col_pos) == 1) {

                    // Eliminate the spillage from the origin cell
                    accessMat(water_level, row_pos, col_pos) -=
                        FIXED(accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);

                    // Compute termination condition: Maximum cell spillage during the iteration
                    if (accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR > max_spillage_iter) {
                        max_spillage_iter = accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
                    }
                    // Statistics: Record maximum cell spillage during the scenario and its time
                    if (accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR > r->max_spillage_scenario) {
                        r->max_spillage_scenario = accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
                        r->max_spillage_minute = *minute;
                    }
                }

                // Accumulate spillage from neighbors
                for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                    int depths = CONTIGUOUS_CELLS;
                    accessMat(water_level, row_pos, col_pos) +=
                        FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
                }
            }
        }

#ifdef DEBUG
#ifndef ANIMATION
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif

        /* Reset ancillary structures */
        for (row_pos = 0; row_pos < rows; row_pos++) {
            for (col_pos = 0; col_pos < columns; col_pos++) {
                for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                    int depths = CONTIGUOUS_CELLS;
                    accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) = 0;
                }
                accessMat(spillage_flag, row_pos, col_pos) = 0;
                accessMat(spillage_level, row_pos, col_pos) = 0;
            }
        }
    }

    cudaDeviceSynchronize();

    r->runtime = get_time() - r->runtime;

    if (p->final_matrix) {
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
    }

    /* Statistics: Total remaining water and maximum amount of water in a cell */
    r->max_water_scenario = 0.0;
    for (row_pos = 0; row_pos < rows; row_pos++) {
        for (col_pos = 0; col_pos < columns; col_pos++) {
            if (FLOATING(accessMat(water_level, row_pos, col_pos)) > r->max_water_scenario)
                r->max_water_scenario = FLOATING(accessMat(water_level, row_pos, col_pos));
            r->total_water += accessMat(water_level, row_pos, col_pos);
        }
    }

    /* Free resources */
    free(ground);
    free(water_level);
    free(spillage_flag);
    free(spillage_level);
    free(spillage_from_neigh);

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
