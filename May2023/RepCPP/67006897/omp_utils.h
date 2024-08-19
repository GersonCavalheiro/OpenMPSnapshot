

#ifndef PROFIT_OMP_UTILS_H_
#define PROFIT_OMP_UTILS_H_

#include "profit/common.h"

namespace profit {


template <typename Callable>
void omp_2d_for(int threads, unsigned int width, unsigned int height, Callable &&f)
{
#if _OPENMP >= 200805 
#pragma omp parallel for collapse(2) schedule(dynamic, 10) if(threads > 1) num_threads(threads)
for (unsigned int j = 0; j < height; j++) {
for (unsigned int i = 0; i < width; i++) {
f(i, j);
}
}
#elif _OPENMP >= 200203 
#pragma omp parallel for schedule(dynamic, 10) if(threads > 1) num_threads(threads)
for (int x = 0; x < int(width * height); x++) {
unsigned int i = x % width;
unsigned int j = x / width;
f(i, j);
}
#else
UNUSED(threads);
for (unsigned int j = 0; j < height; j++) {
for (unsigned int i = 0; i < width; i++) {
f(i, j);
}
}
#endif 
}

}  

#endif 
