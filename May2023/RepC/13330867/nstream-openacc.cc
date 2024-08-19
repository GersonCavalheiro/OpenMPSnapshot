#include <openacc.h>
#include "prk_util.h"
int main(int argc, char * argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/OpenACC STREAM triad: A = B + scalar * C" << std::endl;
int iterations;
size_t length;
try {
if (argc < 3) {
throw "Usage: <# iterations> <vector length>";
}
iterations  = std::atoi(argv[1]);
if (iterations < 1) {
throw "ERROR: iterations must be >= 1";
}
length = std::atol(argv[2]);
if (length <= 0) {
throw "ERROR: vector length must be positive";
}
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations  = " << iterations << std::endl;
std::cout << "Vector length         = " << length << std::endl;
double nstream_time{0};
size_t bytes = length*sizeof(double);
double * RESTRICT A = (double *) acc_malloc(bytes);
double * RESTRICT B = (double *) acc_malloc(bytes);
double * RESTRICT C = (double *) acc_malloc(bytes);
double scalar = 3.0;
{
#pragma acc parallel loop deviceptr(A,B,C)
for (size_t i=0; i<length; i++) {
A[i] = 0.0;
B[i] = 2.0;
C[i] = 2.0;
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) nstream_time = prk::wtime();
#pragma acc parallel loop deviceptr(A,B,C)
for (size_t i=0; i<length; i++) {
A[i] += B[i] + scalar * C[i];
}
}
nstream_time = prk::wtime() - nstream_time;
}
double ar(0);
double br(2);
double cr(2);
for (int i=0; i<=iterations; i++) {
ar += br + scalar * cr;
}
ar *= length;
double asum(0);
#pragma acc parallel loop reduction( +:asum ) deviceptr(A)
for (size_t i=0; i<length; i++) {
asum += prk::abs(A[i]);
}
acc_free(A);
acc_free(B);
acc_free(C);
double epsilon=1.e-8;
if (prk::abs(ar-asum)/asum > epsilon) {
std::cout << "Failed Validation on output array\n"
<< std::setprecision(16)
<< "       Expected checksum: " << ar << "\n"
<< "       Observed checksum: " << asum << std::endl;
std::cout << "ERROR: solution did not validate" << std::endl;
return 1;
} else {
std::cout << "Solution validates" << std::endl;
double avgtime = nstream_time/iterations;
double nbytes = 4.0 * length * sizeof(double);
std::cout << "Rate (MB/s): " << 1.e-6*nbytes/avgtime
<< " Avg time (s): " << avgtime << std::endl;
}
return 0;
}
