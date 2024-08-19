#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "MCMLrng.h"
#define SAFE_PRIMES "safeprimes_base32.txt"
#define SAMPLES (1 << 28)
static unsigned int * rng_const;
static unsigned long long * rng_state;
static void init(void) {
int threads = omp_get_max_threads();
rng_const = malloc(threads * sizeof(unsigned int));
rng_state = malloc(threads * sizeof(unsigned long long));
unsigned long long seed = (unsigned long long) time(NULL);
int error = init_RNG(rng_state, rng_const, threads, SAFE_PRIMES, seed);
assert (error == 0);
}
static void cleanup(void) {
free(rng_const);
free(rng_state);
}
static double pi(void) {
long hits = 0;
float x, y;
int tid;
#pragma omp parallel default(shared) private(x,y,tid) reduction(+:hits)
{
tid = omp_get_thread_num();
unsigned long long rng_state_tid = rng_state[tid];
#pragma omp for schedule(static, 1 << 20)
for (size_t s = 0; s < SAMPLES; ++s) {
x = rand_MWC_co(&rng_state_tid, &rng_const[tid]);
y = rand_MWC_co(&rng_state_tid, &rng_const[tid]);
if (sqrt((double) x * x + y * y) <= 1.0) {
++hits;
}
}
}
return (double)((long double) hits * 4.0 / SAMPLES);
}
int main(int argc, char **argv) {
double t, result;
init();
t = omp_get_wtime();
result = pi();
t = omp_get_wtime() - t;
printf("Pi: %.8f, %.4fs\n", result, t);
cleanup();
return 0;
}
