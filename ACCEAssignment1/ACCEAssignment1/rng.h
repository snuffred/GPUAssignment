/*
 * Simple random generator
 * LCG (Linear Congruential Generator)
 *
 * Reference sequential version (Do not modify this code)
 *
 * Adapted for the ACCE course (XM_0171) at VU Amsterdam, Period 5 2025-2026.
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */
#ifndef _RNG_H_
#define _RNG_H_

#include <math.h>
#include <stdint.h>

/*
 * Constants
 */
#define RNG_MULTIPLIER 6364136223846793005ULL
#define RNG_INCREMENT 1442695040888963407ULL

/*
 * Type for random sequences state
 */
typedef uint64_t rng_t;

/*
 * Constructor: Create a new state from a seed
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    rng_t rng_new(uint64_t seed);

/*
 * Next: Advance state and return a double number uniformely distributed
 * Adapted from the implementation on PCG (https://www.pcg-random.org/)
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    double rng_next(rng_t *seq);

/*
 * Next: Advance state and return a double number uniformely distributed between limits
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    double rng_next_between(rng_t *seq, double min, double max);

/*
 * Next Normal: Advance state and return a double number distributed with a normal(mu,sigma)
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    double rng_next_normal(rng_t *seq, double mu, double sigma);

/*
 * Skip ahead: Advance state with an arbitrary jump in log time
 * Adapted from the implementation on PCG (https://www.pcg-random.org/)
 */
#ifdef __CUDACC__
__host__ __device__
#endif
    void rng_skip(rng_t *seq, uint64_t steps);

#endif
