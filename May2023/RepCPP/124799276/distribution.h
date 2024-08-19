#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cmath>
#include <omp.h>
const float delta_max = 1.0f;
#pragma omp declare simd
float dist_func(const float alpha, const float rn);
#endif
