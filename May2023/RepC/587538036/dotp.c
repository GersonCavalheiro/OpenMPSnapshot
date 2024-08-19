#include <dotp.h>
double dotp(double* x, double* y) {
double global_sum = 0.0;
#pragma omp parallel
{
#pragma omp for
for(int i=0; i<ARRAY_SIZE; i++) {
#pragma omp critical
global_sum += x[i] * y[i];
}
}
return global_sum;
}
