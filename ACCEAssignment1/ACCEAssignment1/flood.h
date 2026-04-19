#ifndef FLOODH
#define FLOODH

#include <math.h>
#include <sys/time.h>

/*
 * Water levels are stored with fixed precision. Fixed precision stores real numbers
 * as scaled integers (e.g., 1.23 -> 123 with a fixed scale).
 * Using fixed precision we can reduce result differences when arithmetic operations are reordered.
 */
#define PRECISION 1000000
#define FIXED(a) ((int)((a) * PRECISION))
#define FLOATING(a) ((float)(a) / PRECISION)
#define PRECISION_FIXED 1
#define PRECISION_FLOAT 2

/*
 * Scenario size (km x km)
 */
#define SCENARIO_SIZE 30

/*
 * Spillage factor for equilibrium
 */
#define SPILLAGE_FACTOR 2

/*
 * Structure to represent moving rainy clouds
 * This structure can be changed and/or optimized by the students
 */
typedef struct {
    float x;         // x coordinate of the center
    float y;         // y coordinate of the center
    float radius;    // radius of the cloud (km)
    float intensity; // rainfall intensity (cm/h)
    float speed;     // speed of movement (km/h)
    float angle;     // angle of movement
    int active;      // active cloud
} Cloud_t;

struct parameters {
    int rows;
    int columns;
    char ground_scenario;
    float threshold;
    int num_minutes;
    float ex_factor;
    int num_clouds;
    float *ground;
    Cloud_t *clouds;
    int final_matrix;
};

struct results {
    int minute;
    float max_water_scenario;
    double max_spillage_scenario;
    int max_spillage_minute;
    double runtime;
    // Metrics to accumulate fixed point values
    long total_water;
    long total_water_loss;
    long total_rain;
};

#ifdef __CUDACC__
extern "C" {
#endif
double get_time();

void print_matrix(int precision_type, int rows, int columns, void *mat, const char *msj);
#ifdef DEBUG
void print_clouds(int num_clouds, Cloud_t *clouds);
#endif // DEBUG
#ifdef __CUDACC__
}
#endif

/*
 *
 * The following components may be modified for flood_cuda.cu
 *
 */

/*
 * Utils: Number of contiguous cells to consider for water spillage
 * 	0: up, 1: down, 2: left, 3: right
 * 	Displacements for the contiguous cells
 * 	This data structure can be changed and/or optimized by the students
 */
#define CONTIGUOUS_CELLS 4
#ifdef __CUDACC__
__device__
#endif
    static int displacements[CONTIGUOUS_CELLS][2] = {
        {-1, 0}, // Top
        {1, 0},  // Bottom
        {0, -1}, // Left
        {0, 1}   // Right
};

/*
 * Utils: Macro-functions to transform coordinates, from scenario to matrix cells, and back
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define COORD_SCEN2MAT_X(x) (x * columns / SCENARIO_SIZE)
#define COORD_SCEN2MAT_Y(y) (y * rows / SCENARIO_SIZE)
#define COORD_MAT2SCEN_X(c) (c * SCENARIO_SIZE / columns)
#define COORD_MAT2SCEN_Y(r) (r * SCENARIO_SIZE / rows)

/*
 * Utils: Macro functions for the min and max of two numbers
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Utils: Macro function to simplify accessing data of 2D and 3D matrixes stored in a flattened array
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define accessMat(arr, exp1, exp2) arr[(int)(exp1) * columns + (int)(exp2)]
#define accessMat3D(arr, exp1, exp2, exp3) arr[((int)(exp1) * columns * depths) + ((int)(exp2) * depths) + (int)(exp3)]

#endif