

#ifndef _PCR_SMALL_SYSTEMS_
#define _PCR_SMALL_SYSTEMS_

#include <omp.h>
#include "tridiagonal.h"
#include "pcr_kernels.cpp"

const char *pcrKernelNames[] = { 
"pcr_small_systems_kernel",    
"pcr_branch_free_kernel",      
};  

double pcr_small_systems(float *a, float *b, float *c, float *d, float *x, 
int system_size, int num_systems, int id = 0)
{
shrLog(" %s\n", pcrKernelNames[id]);
double sum_time;
const unsigned int mem_size = num_systems * system_size;

#pragma omp target data map(to: a[0:mem_size], \
b[0:mem_size], \
c[0:mem_size], \
d[0:mem_size]) \
map(from: x[0:mem_size])
{

size_t szTeams;
size_t szThreads;
int iterations = my_log2 (system_size/2);

szThreads = system_size;
szTeams = num_systems; 

if (id == 0)
pcr_small_systems_kernel(
a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
else
pcr_branch_free_kernel(
a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);

shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

shrDeltaT(0);
for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
{
if (id == 0)
pcr_small_systems_kernel(
a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
else
pcr_branch_free_kernel(
a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
}
sum_time = shrDeltaT(0);
}

double time = sum_time / BENCH_ITERATIONS;
return time;
}
#endif
