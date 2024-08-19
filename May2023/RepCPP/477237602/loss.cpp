
#include "neural.hpp"



double nn::mse_loss(double* (&Y), int dim)
{
double l = 0.0;                                                         
#pragma omp simd reduction(+ : l)
for (int i = 0; i < dim; i += 1)
{
l += (1.0 / 2.0) * (Y[i] - a[layers.size() - 1][i]) * (Y[i] - a[layers.size() - 1][i]);
}

return l;
}
