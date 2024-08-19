#include <mkl.h>
#include <cstdio>
#include "distribution.h"


int diffusion(const int n_particles,
const int n_steps,
const float x_threshold,
const float alpha,
VSLStreamStatePtr rnStream) {
int n_escaped=0;
float positions [n_particles];

for (int j = 0; j < n_steps; j++) {
float rn;
vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 1, &rn, -1.0, 1.0);
for (int i = 0; i < n_particles; i++) {
positions[i] += dist_func(alpha, rn);
}
}

#pragma omp simd
for (int i = 0; i < n_particles; i++) {
if (positions[i] > x_threshold) n_escaped++;
}

return n_escaped;
}
