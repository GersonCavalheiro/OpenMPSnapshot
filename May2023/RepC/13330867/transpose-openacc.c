#include <openacc.h>
#include "prk_util.h"
int main(int argc, char * argv[])
{
printf("Parallel Research Kernels version %d\n", PRKVERSION );
printf("C11/OpenACC Matrix transpose: B = A^T\n");
if (argc < 3) {
printf("Usage: <# iterations> <matrix order> [tile size]\n");
return 1;
}
int iterations = atoi(argv[1]);
if (iterations < 1) {
printf("ERROR: iterations must be >= 1\n");
return 1;
}
int order = atoi(argv[2]);
if (order <= 0) {
printf("ERROR: Matrix Order must be greater than 0\n");
return 1;
}
int tile_size = (argc>3) ? atoi(argv[3]) : 32;
if (tile_size <= 0) tile_size = order;
printf("Number of iterations  = %d\n", iterations);
printf("Matrix order          = %d\n", order);
#ifdef __GNUC__
printf("Tile size             = %s\n", "automatic (GCC)");
#else
printf("Tile size             = %d\n", tile_size);
#endif
double trans_time = 0.0;
size_t bytes = order*order*sizeof(double);
double * restrict A = acc_malloc(bytes);
double * restrict B = acc_malloc(bytes);
{
#pragma acc parallel loop deviceptr(A,B)
for (int i=0;i<order; i++) {
for (int j=0;j<order;j++) {
A[i*order+j] = (double)(i*order+j);
B[i*order+j] = 0.0;
}
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) trans_time = prk_wtime();
#ifdef __GNUC__
#pragma acc parallel loop tile(*,*) deviceptr(A,B)
#else
#pragma acc parallel loop tile(tile_size,tile_size) deviceptr(A,B)
#endif
for (int i=0;i<order; i++) {
for (int j=0;j<order;j++) {
B[i*order+j] += A[j*order+i];
A[j*order+i] += 1.0;
}
}
}
trans_time = prk_wtime() - trans_time;
}
const double addit = (iterations+1.) * (iterations/2.);
double abserr = 0.0;
#pragma acc parallel loop reduction( +:abserr ) deviceptr(B)
for (int j=0; j<order; j++) {
for (int i=0; i<order; i++) {
const size_t ij = i*order+j;
const size_t ji = j*order+i;
const double reference = (double)(ij)*(1.+iterations)+addit;
abserr += fabs(B[ji] - reference);
}
}
acc_free(A);
acc_free(B);
#ifdef VERBOSE
printf("Sum of absolute differences: %lf\n", abserr);
#endif
const double epsilon = 1.0e-8;
if (abserr < epsilon) {
printf("Solution validates\n");
const double avgtime = trans_time/iterations;
printf("Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * bytes/avgtime, avgtime );
} else {
printf("ERROR: Aggregate squared error %lf exceeds threshold %lf\n", abserr, epsilon );
return 1;
}
return 0;
}
