#include "openmp_integral_image.h"
void transpose(unsigned long *src, unsigned long *dst, const int N, const int M) {
#pragma omp parallel for
for(int n = 0; n<N*M; n++) {
int i = n/N;
int j = n%N;
dst[n] = src[M*j + i];
}
}
unsigned long * integralImageMP(unsigned long*x, int n, int m, int threads){
unsigned long * out = new unsigned long[n*m];
unsigned long * rows = new unsigned long[n*m];
if(threads != 0){
omp_set_dynamic(0);     
omp_set_num_threads(threads); 
}
#pragma omp parallel for
for (int i = 0; i < n; ++i)
{
rows[i*m] = x[i*m];
for (int j = 1; j < m; ++j)
{
rows[i*m + j] = x[i*m + j] + rows[i*m + j - 1];
}
}
transpose(rows, out, n, m);
#pragma omp parallel for
for (int i = 0; i < m; ++i)
{
rows[i*n] = out[i*n];
for (int j = 1; j < n; ++j)
{
rows[i*n + j] = out[i*n + j] + rows[i*n + j - 1];
}
}
transpose(rows, out, m, n);
delete [] rows;
return out;
}