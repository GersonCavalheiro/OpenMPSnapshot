#ifndef L_H
#define L_H
#include <cmath>

#pragma omp declare simd
float L(const float alpha, const float phase, const float x);
#endif