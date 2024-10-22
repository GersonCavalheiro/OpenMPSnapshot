

#ifndef _SWEEP_SMALL_SYSTEMS_
#define _SWEEP_SMALL_SYSTEMS_

#include <omp.h>
#include <algorithm>   
#include "tridiagonal.h"
#include "sweep_kernels.cpp"


const char *sweepKernelNames[] = { 
"sweep_small_systems_local_kernel",      
"sweep_small_systems_global_kernel",    
"sweep_small_systems_global_vec4_kernel",  
"transpose",              
};  

double runReorderKernel(float* d_a, float* d_t, int width, int height)
{
size_t szTeamX = shrRoundUp(TRANSPOSE_BLOCK_DIM, width) / TRANSPOSE_BLOCK_DIM;
size_t szTeamY = shrRoundUp(TRANSPOSE_BLOCK_DIM, height) / TRANSPOSE_BLOCK_DIM;
size_t szTeam = szTeamX * szTeamY;

transpose(d_t, d_a, width, height, szTeamX, szTeam);

double sum_time = 0.0;
shrDeltaT(0);
for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
{
transpose(d_t, d_a, width, height, szTeamX, szTeam);
}

sum_time = shrDeltaT(0);
double time = sum_time / BENCH_ITERATIONS;

return time;
}

double runSweepKernel(
float* a, 
float* b, 
float* c, 
float* d, 
float* x, 
float* t, 
float* w, 
int system_size, 
int num_systems,
bool reorder)
{
size_t szTeam;
size_t szThreads;

if (useVec4) szThreads = SWEEP_BLOCK_SIZE / 4;
else szThreads = SWEEP_BLOCK_SIZE;
szTeam = shrRoundUp(SWEEP_BLOCK_SIZE, num_systems) / szThreads;

if (useLmem) 
sweep_small_systems_local_kernel(
a, b, c, d, x, system_size, num_systems, reorder, szTeam, szThreads);
else if (useVec4) 
sweep_small_systems_global_vec4_kernel(
a, b, c, d, x, w, system_size, num_systems, reorder, szTeam, szThreads);
else 
sweep_small_systems_global_kernel(
a, b, c, d, x, w, system_size, num_systems, reorder, szTeam, szThreads);

shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

double sum_time = 0.0;
shrDeltaT(0);
for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
{
if (useLmem) 
sweep_small_systems_local_kernel(
a, b, c, d, x, system_size, num_systems, reorder, szTeam, szThreads);
else if (useVec4) 
sweep_small_systems_global_vec4_kernel(
a, b, c, d, x, w, system_size, num_systems, reorder, szTeam, szThreads);
else 
sweep_small_systems_global_kernel(
a, b, c, d, x, w, system_size, num_systems, reorder, szTeam, szThreads);
}
sum_time = shrDeltaT(0);
double time = sum_time / BENCH_ITERATIONS;

return time;
}

double sweep_small_systems(float *a, float *b, float *c, float *d, float *x, 
int system_size, int num_systems, bool reorder = false)
{
if (reorder) shrLog("sweep_data_reorder_kernel\n"); 
if (useLmem) shrLog("%s\n", sweepKernelNames[0]); 
else if (useVec4) shrLog("%s\n", sweepKernelNames[2]); 
else shrLog("%s\n", sweepKernelNames[1]); 

const unsigned int mem_size = num_systems * system_size;

float* t = (float*) malloc (mem_size * sizeof(float));
float* w = (float*) malloc (mem_size * sizeof(float));

int workSize = num_systems;

double reorder_time = 0.0;
double solver_time = 0.0;

#pragma omp target data map(to: a[0:mem_size], \
b[0:mem_size], \
c[0:mem_size], \
d[0:mem_size]) \
map(alloc: x[0:mem_size], \
t[0:mem_size], \
w[0:mem_size])
{
if (reorder)
{
reorder_time += runReorderKernel(a, t, system_size, workSize);
std::swap(a, t);

reorder_time += runReorderKernel(b, t, system_size, workSize);
std::swap(b, t);

reorder_time += runReorderKernel(c, t, system_size, workSize);
std::swap(c, t);

reorder_time += runReorderKernel(d, t, system_size, workSize);
std::swap(d, t);
}

solver_time = runSweepKernel(a, b, c, d, x, t, w, system_size, workSize, reorder);

if (reorder)
{
reorder_time += runReorderKernel(x, t, workSize, system_size);
std::swap(x, t);
} 
#pragma omp target update from (x[0:mem_size])
}

if (!reorder) free(t);
free(w);
return solver_time + reorder_time;
}

#endif
