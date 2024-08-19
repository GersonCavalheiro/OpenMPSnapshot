#include <mkl.h>
#include "distribution.h"


#pragma omp declare simd
int diffusion(const int n_particles, 
const int n_steps, 
const float x_threshold,
const float alpha, 
VSLStreamStatePtr rnStream) {
int n_escaped=0;


float rn[n_particles];
vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, rn, -1.0, 1.0);

float x_pos[n_particles];
x_pos[0:n_particles] = 0.0f;

for (int j=0; j<n_steps; j++)
{
for (int i=0; i<n_particles; i++)
{
x_pos[i] += dist_func(alpha, rn[i]);
}
}


for (int i=0; i<n_particles; i++)
{
if (x_pos[i] > x_threshold) n_escaped++;
}

return n_escaped;
}