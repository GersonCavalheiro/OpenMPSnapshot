#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "MCMLrng.h"
#define SAFE_PRIMES "safeprimes_base32.txt"
#define N_RNGS (1 << 10)
#define SAMPLES (1 << 18)
static float sum[N_RNGS];
static unsigned int rng_const[N_RNGS];
static unsigned long long rng_state[N_RNGS];
static void init_rng(void) {
unsigned long long seed = (unsigned long long) time(NULL);
int error = init_RNG(rng_state, rng_const, N_RNGS, SAFE_PRIMES, seed);
assert (error == 0);
}
static void sequential(void) {
for (size_t s = 0; s < SAMPLES; ++s) {
for (size_t i = 0; i < N_RNGS; ++i) {
float r = rand_MWC_co(&rng_state[i], &rng_const[i]);
sum[i] += r / SAMPLES;
}
}
}
static void parallel(void) {
#pragma omp parallel
for (size_t s = 0; s < SAMPLES; ++s) {
#pragma omp for
for (size_t i = 0; i < N_RNGS; ++i) {
float r = rand_MWC_co(&rng_state[i], &rng_const[i]);
sum[i] += r / SAMPLES;
}
}
}
static float avg(void) {
float tmp = 0.0f;
for (size_t i = 0; i < N_RNGS; ++i) {
tmp += sum[i];
}
return tmp / N_RNGS;
}
int main(int argc, char **argv) {
double seq_t, par_t;
init_rng();
memset(sum, 0, N_RNGS * sizeof(float));
seq_t = omp_get_wtime();
sequential();
seq_t = omp_get_wtime() - seq_t;
printf("Seq: %.4fs, %.4f avg\n", seq_t, avg());
memset(sum, 0, N_RNGS * sizeof(float));
par_t = omp_get_wtime();
parallel();
par_t = omp_get_wtime() - par_t;
printf("Parallel: %.4fs, %.4f avg, %.2f speedup, %.4f efficiency\n", par_t, avg(), seq_t / par_t, seq_t / (omp_get_max_threads() * par_t));
return 0;
}
