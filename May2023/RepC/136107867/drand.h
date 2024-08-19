#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
static uint64_t seed;
#pragma omp threadprivate(seed)
void dsrand(unsigned s);
void dsrand_parallel(unsigned s);
double drand(void);
