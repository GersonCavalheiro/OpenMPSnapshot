

#include "sobol.h"
#include "sobol_gpu.h"

#define k_2powneg32 2.3283064E-10F


#pragma omp declare target
int _ffs(const int x) {
for (int i = 0; i < 32; i++)
if ((x >> i) & 1) return (i+1);
return 0;
};
#pragma omp end declare target

double sobolGPU(int repeat, int n_vectors, int n_dimensions, 
unsigned int *dir, float *out)
{
const int threadsperblock = 64;

size_t dimGrid_y = n_dimensions;
size_t dimGrid_x;

if (n_dimensions < (4 * 24))
{
dimGrid_x = 4 * 24;
}
else
{
dimGrid_x = 1;
}

if (dimGrid_x > (unsigned int)(n_vectors / threadsperblock))
{
dimGrid_x = (n_vectors + threadsperblock - 1) / threadsperblock;
}

unsigned int targetDimGridX = dimGrid_x;

for (dimGrid_x = 1 ; dimGrid_x < targetDimGridX ; dimGrid_x *= 2);

size_t numTeam =  dimGrid_x * dimGrid_y;

auto start = std::chrono::steady_clock::now();

for (int i = 0; i < repeat; i++) {
#pragma omp target teams num_teams(numTeam) thread_limit(threadsperblock)
{
unsigned int v[n_directions];
#pragma omp parallel 
{
unsigned int teamX = omp_get_team_num() % dimGrid_x;
unsigned int teamY = omp_get_team_num() / dimGrid_x; 
unsigned int tidX = omp_get_thread_num();
unsigned int threadSizeX = omp_get_num_threads();

dir += n_directions * teamY;
out += n_vectors * teamY;

if (tidX < n_directions)
{
v[tidX] = dir[tidX];
}

#pragma omp barrier

int i0     = teamX * threadSizeX + tidX;
int stride = dimGrid_x * threadSizeX;

unsigned int g = i0 ^ (i0 >> 1);

unsigned int X = 0;
unsigned int mask;

for (unsigned int k = 0 ; k < _ffs(stride) - 1 ; k++)
{
mask = - (g & 1);
X ^= mask & v[k];
g = g >> 1;
}

if (i0 < n_vectors)
{
out[i0] = (float)X * k_2powneg32;
}

unsigned int v_log2stridem1 = v[_ffs(stride) - 2];
unsigned int v_stridemask = stride - 1;

for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
{
X ^= v_log2stridem1 ^ v[_ffs(~((i - stride) | v_stridemask)) - 1];
out[i] = (float)X * k_2powneg32;
}

}
}
}

auto end = std::chrono::steady_clock::now();
double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
return time;
}
