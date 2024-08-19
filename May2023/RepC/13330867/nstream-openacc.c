#include <openacc.h>
#include "prk_util.h"
int main(int argc, char * argv[])
{
printf("Parallel Research Kernels version %d\n", PRKVERSION );
printf("C11/OpenACC STREAM triad: A = B + scalar * C\n");
if (argc < 3) {
printf("Usage: <# iterations> <vector length>\n");
return 1;
}
int iterations = atoi(argv[1]);
if (iterations < 1) {
printf("ERROR: iterations must be >= 1\n");
return 1;
}
size_t length = atol(argv[2]);
if (length <= 0) {
printf("ERROR: Vector length must be greater than 0\n");
return 1;
}
printf("Number of iterations = %d\n", iterations);
printf("Vector length        = %zu\n", length);
double nstream_time = 0.0;
size_t bytes = length*sizeof(double);
double * restrict A = acc_malloc(bytes);
double * restrict B = acc_malloc(bytes);
double * restrict C = acc_malloc(bytes);
double scalar = 3.0;
{
#pragma acc parallel loop deviceptr(A,B,C)
for (size_t i=0; i<length; i++) {
A[i] = 0.0;
B[i] = 2.0;
C[i] = 2.0;
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) nstream_time = prk_wtime();
#pragma acc parallel loop deviceptr(A,B,C)
for (size_t i=0; i<length; i++) {
A[i] += B[i] + scalar * C[i];
}
}
nstream_time = prk_wtime() - nstream_time;
}
double ar = 0.0;
double br = 2.0;
double cr = 2.0;
for (int i=0; i<=iterations; i++) {
ar += br + scalar * cr;
}
ar *= length;
double asum = 0.0;
#pragma acc parallel loop reduction( +:asum ) deviceptr(A)
for (size_t i=0; i<length; i++) {
asum += fabs(A[i]);
}
acc_free(A);
acc_free(B);
acc_free(C);
double epsilon=1.e-8;
if (fabs(ar-asum)/asum > epsilon) {
printf("Failed Validation on output array\n"
"       Expected checksum: %lf\n"
"       Observed checksum: %lf\n"
"ERROR: solution did not validate\n", ar, asum);
return 1;
} else {
printf("Solution validates\n");
double avgtime = nstream_time/iterations;
double nbytes = 4.0 * length * sizeof(double);
printf("Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
}
return 0;
}
