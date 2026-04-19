/*
 * Simulation of rainwater flooding
 *
 * Framework code. In principle, this file should not be modified as the current implementation
 * is sufficient for completing the assignment. However, you are allowed to make changes if needed to
 * support your optimizations (check the README).
 *
 * Adapted for the ACCE course (XM_0171) at VU Amsterdam, Period 5 2025-2026.
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "flood.h"

/*
 * Utils: Random generator
 */
#include "rng.h"

void do_compute(const struct parameters *p, struct results *r);

/*
 * Function: Generate ground height for a given position
 *  This function can be optimized by the students, but the
 *  output must be unchanged.
 */
float get_height(char scenario, int row, int col, int rows, int columns) {
    // Choose scenario limits
    float x_min, x_max, y_min, y_max;
    if (scenario == 'M') { // Mountains scenario
        x_min = -3.3;
        x_max = 5.1;
        y_min = -0.5;
        y_max = 8.8;
    } else { // Valley scenarios
        x_min = -5.5;
        x_max = -3;
        y_min = -0.1;
        y_max = 4.2;
    }

    // Compute scenario coordinates of the cell position
    float x = x_min + ((x_max - x_min) / columns) * col;
    float y = y_min + ((y_max - y_min) / rows) * row;

    // Compute function height
    float height = -1 / (x * x + 1) + 2 / (y * y + 1) + 0.5 * sin(5 * sqrt(x * x + y * y)) / sqrt(x * x + y * y) +
                   (x + y) / 3 + sin(x) * cos(y) + 0.4 * sin(3 * x + y) + 0.25 * cos(4 * y + x);

// Substitute by the dam height in the proper scenarios
#define LOW_DAM_HEIGHT -1.0
#define HIGH_DAM_HEIGHT -0.4
    if (scenario == 'D' && x <= -4.96 && x >= -5.0) {
        if (height < HIGH_DAM_HEIGHT) {
            height = HIGH_DAM_HEIGHT;
        }
    } else if (scenario == 'd' && x <= -5.3 && x >= -5.34) {
        if (height < LOW_DAM_HEIGHT) {
            height = LOW_DAM_HEIGHT;
        }
    }

    // Transform to meters
    if (scenario == 'M')
        return height * 30 + 400;
    else
        return height * 20 + 100;
}

/*
 * Utils: Function to get wall time
 */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Function: Print the current state of the simulation
 */
void print_matrix(int precision_type, int rows, int columns, void *mat, const char *msj) {
    /*
     * You don't need to optimize this function, it is only for pretty
     * printing and debugging purposes.
     */
    int i, j;
#ifndef ANIMATION
    printf("%s:\n", msj);
    printf("+");
    for (j = 0; j < columns; j++)
        printf("----------");
    printf("+\n");
    printf("\n");
#endif
    // Y coordinates: Reversed, higher rows up
    for (i = rows - 1; i >= 0; i--) {
#ifndef ANIMATION
        printf("|");
#endif
        // X coordinates
        for (j = 0; j < columns; j++) {
            if (precision_type == PRECISION_FLOAT)
                printf(" %10.4f", accessMat(((float *)mat), i, j));
            else
                printf(" %10.4f", FLOATING(accessMat(((int *)mat), i, j)));
        }
#ifndef ANIMATION
        printf("|\n");
#endif
        printf("\n");
    }
#ifndef ANIMATION
    printf("+");
    for (j = 0; j < columns; j++)
        printf("----------");
    printf("+\n\n\n");
#else
    printf("\n");
#endif
}

#ifdef DEBUG

/*
 * Function: Print the current state of the clouds
 */
void print_clouds(int num_clouds, Cloud_t *clouds) {
    /*
     * You don't need to optimize this function, it is only for pretty
     * printing and debugging purposes.
     */
    printf("Clouds:\n");
    for (int i = 0; i < num_clouds; i++) {
        printf("Cloud %d: x = %f, y = %f, radius = %f, intensity = %f, speed = %f, angle = %f\n", i, clouds[i].x,
               clouds[i].y, clouds[i].radius, clouds[i].intensity, clouds[i].speed, clouds[i].angle);
    }
    printf("\n");
}
#endif // DEBUG

/*
 * Function: Print the program usage line in stderr
 */
void show_usage(char *program_name) {
    fprintf(stderr, "\nFlood Simulation - Simulate rain and flooding in %d x %d km^2\n", SCENARIO_SIZE, SCENARIO_SIZE);
    fprintf(stderr, "----------------------------------------------------------------\n");
    fprintf(stderr, "Usage: %s ", program_name);
    fprintf(stderr, "<rows> <columns> <ground_scenario(M|V|D|d)> <threshold> <num_minutes> <exaggeration_factor> "
                    "<front_distance> <front_width> <front_depth> <front_direction(grad.)>\n");
    fprintf(stderr, "<num_random_clouds> <cloud_max_radius(km)> <cloud_max_intensity(mm/h)> <cloud_max_speed(km/h)> "
                    "<cloud_max_angle_aperture(grad.)> <clouds_rnd_seed>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\tOptional arguments for special clouds: <cloud_start_x(km)> <cloud_start_y(km)> "
                    "<cloud_radius(km)> <cloud_intensity(mm/h)> <cloud_speed(km/h)> <cloud_angle(grad.)> ...\n");
    fprintf(stderr, "\n");
    fprintf(stderr,
            "\tGround models: 'M' mountain lakes, 'V' valley, 'D' valley with high dam, 'd' valley with low dam\n");
    fprintf(stderr, "\tIntensity of rain (mm/h): Strong (15-30), Very strong (30-60), Torrential: Above 60\n");
    fprintf(stderr, "\n");
}

/*
 * Function: Initialize cloud with random values
 *  This function can be optimized by the students, but the
 *  output must be unchanged.
 */
Cloud_t cloud_init(Cloud_t cloud_model, float front_distance, float front_width, float front_depth,
                   float front_direction, int rows, int cols, rng_t *rnd_state) {
    Cloud_t cloud;

    // Random position around the front center
    cloud.x = (float)rng_next_between(rnd_state, 0, front_width) - front_width / 2;
    cloud.y = (float)rng_next_between(rnd_state, 0, front_depth) - front_depth / 2;

    // Rotate
    float opposite = front_direction + 180;
    float tmp_x = cloud.x;
    float tmp_y = cloud.y;
    cloud.x = tmp_x * cos(opposite * M_PI / 180.0) - tmp_y * sin(opposite * M_PI / 180.0);
    cloud.y = tmp_x * sin(opposite * M_PI / 180.0) + tmp_y * cos(opposite * M_PI / 180.0);

    // Move center
    float x_center = front_distance * cos(opposite * M_PI / 180.0) + SCENARIO_SIZE / 2;
    float y_center = front_distance * sin(opposite * M_PI / 180.0) + SCENARIO_SIZE / 2;
    cloud.x += x_center;
    cloud.y += y_center;

    // Cloud random parameters
    cloud.radius = (float)rng_next_between(rnd_state, cloud_model.radius / 2, cloud_model.radius);
    cloud.intensity = (float)rng_next_between(rnd_state, cloud_model.intensity / 2, cloud_model.intensity);
    cloud.speed = (float)rng_next_between(rnd_state, cloud_model.speed / 2, cloud_model.speed);
    cloud.angle = front_direction + (float)rng_next_between(rnd_state, 0, cloud_model.angle) - cloud_model.angle / 2;
    cloud.active = 1;
    return cloud;
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
#ifdef DEBUG
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#endif

#define NUM_FIXED_ARGS 17

    struct parameters p;
    p.final_matrix = 0;

    /* 1. Read simulation arguments */
    /* 1.1. Check minimum number of arguments */
    if (argc < NUM_FIXED_ARGS) {
        fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (argc > NUM_FIXED_ARGS) {
        if ((argc - NUM_FIXED_ARGS) % 6 != 0) {
            fprintf(stderr,
                    "-- Error: Wrong number of arguments, there should be %d compulsory arguments + groups of 6 "
                    "optional arguments\n",
                    NUM_FIXED_ARGS);
            exit(EXIT_FAILURE);
        }
    }

    /* 1.2. Read ground sizes and selection of ground scenario */
    p.rows = atoi(argv[1]);
    p.columns = atoi(argv[2]);
    p.ground_scenario = argv[3][0];
    if (p.ground_scenario != 'M' && p.ground_scenario != 'V' && p.ground_scenario != 'D' && p.ground_scenario != 'd') {
        fprintf(stderr, "-- Error: Wrong ground scenario\n\n");
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* 1.3. Read termination conditions */
    p.threshold = atof(argv[4]);
    p.num_minutes = atoi(argv[5]);

    /* 1.4. Read clouds data */
    p.ex_factor = atoi(argv[6]);
    float front_distance = atof(argv[7]);
    float front_width = atof(argv[8]);
    float front_depth = atof(argv[9]);
    float front_direction = atof(argv[10]);
    p.num_clouds = atoi(argv[11]);
    Cloud_t cloud_model;
    cloud_model.x = cloud_model.y = 0;
    cloud_model.radius = atof(argv[12]);
    cloud_model.intensity = atof(argv[13]);
    cloud_model.speed = atof(argv[14]);
    cloud_model.angle = atof(argv[15]);
    cloud_model.active = 0;
    unsigned int seed_clouds = (unsigned int)atol(argv[16]);
    // Initialize random sequence
    rng_t rnd_state = rng_new(seed_clouds);

    /* 1.5. Read the non-random clouds information */
    int num_clouds_arg = (argc - NUM_FIXED_ARGS) / 6;
    Cloud_t clouds_arg[num_clouds_arg];
    int idx;
    for (idx = NUM_FIXED_ARGS; idx < argc; idx += 6) {
        int pos = (idx - NUM_FIXED_ARGS) / 6;
        clouds_arg[pos].x = atof(argv[idx]);
        clouds_arg[pos].y = atof(argv[idx + 1]);
        clouds_arg[pos].radius = atof(argv[idx + 2]);
        clouds_arg[pos].intensity = atof(argv[idx + 3]);
        clouds_arg[pos].speed = atof(argv[idx + 4]);
        clouds_arg[pos].angle = atof(argv[idx + 5]);
    }

#ifdef DEBUG
#ifdef ANIMATION
    printf("%d %d\n", p.rows, p.columns);
    printf("%d\n", p.num_minutes + 1);
#else
    printf("Arguments, Num_minutes: %d\n", p.num_minutes);
    printf("Arguments, Rows: %d, Columns: %d\n", p.rows, p.columns);
    printf("Arguments, Groud scenario: %c\n", p.ground_scenario);
    printf("Arguments, Num_clouds: %d, Max_radius: %f, Max_intensity: %f, Max_speed: %f, Max_angle: %f, seed: %u\n",
           p.num_clouds, cloud_model.radius, cloud_model.intensity, cloud_model.speed, cloud_model.angle, seed_clouds);
    for (idx = 0; idx < num_clouds_arg; idx++) {
        printf("Arguments, Optional cloud %d: x: %f, y: %f, Radius: %f, Intensity: %f, Speed: %f, Angle: %f\n", idx,
               clouds_arg[idx].x, clouds_arg[idx].y, clouds_arg[idx].radius, clouds_arg[idx].intensity,
               clouds_arg[idx].speed, clouds_arg[idx].angle);
    }
    printf("\n");
#endif
#endif

    /* Initialize clouds */

    Cloud_t *clouds;
    int rows = p.rows, columns = p.columns;
    float *ground = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)columns);
    for (int row_pos = 0; row_pos < rows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            accessMat(ground, row_pos, col_pos) = get_height(p.ground_scenario, row_pos, col_pos, rows, columns);
        }
    }
    p.ground = ground;

    clouds = (Cloud_t *)malloc(sizeof(Cloud_t) * (p.num_clouds + num_clouds_arg));

    if (clouds == NULL) {
        fprintf(stderr, "-- Error allocating clouds structures for size: %d\n", p.num_clouds);
        exit(EXIT_FAILURE);
    }

    int cloud;
    for (cloud = 0; cloud < p.num_clouds; cloud++) {
        clouds[cloud] = cloud_init(cloud_model, front_distance, front_width, front_depth, front_direction, p.rows,
                                   p.columns, &rnd_state);
    }
    for (cloud = 0; cloud < num_clouds_arg; cloud++)
        clouds[p.num_clouds + cloud] = clouds_arg[cloud];
    p.num_clouds += num_clouds_arg;
    p.clouds = clouds;

#ifdef ANIMATION
    // Animation mode always generate the data for all the iterations
    p.threshold = -INFINITY;
#endif

    /* Initialize metrics */
    struct results r = {.minute = 0,
                        .max_water_scenario = 0.0,
                        .max_spillage_scenario = 0.0,
                        .max_spillage_minute = 0,
                        .runtime = 0,
                        .total_water = 0,
                        .total_water_loss = 0,
                        .total_rain = 0};

    do_compute(&p, &r);

    /* Free resources */
    free(clouds);

#ifndef ANIMATION
    /* Output stats */
    printf("\nTime: %lf\n", r.runtime);

    printf("Result: %d, %d, %10.6lf, %10.6lf, %10.6lf, %10.6lf, %10.6f\n\n", r.minute, r.max_spillage_minute,
           r.max_spillage_scenario, r.max_water_scenario, FLOATING(r.total_rain), FLOATING(r.total_water),
           FLOATING(r.total_water_loss));
    printf("Check precision loss: %10.6f\n\n",
           FLOATING(r.total_rain) - FLOATING(r.total_water) - FLOATING(r.total_water_loss));
#endif

    return 0;
}
