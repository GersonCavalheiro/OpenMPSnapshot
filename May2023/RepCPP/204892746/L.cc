#include "L.h"

#pragma omp declare simd
float L(const float alpha, const float phase, const float x) {
return expf(-alpha*(x-phase)*(x-phase));
}