#include <openacc.h>
#include "prk_util.h"
int main(int argc, char * argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/OpenMP TARGET Matrix transpose: B = A^T" << std::endl;
int iterations;
int order;
int tile_size;
try {
if (argc < 3) {
throw "Usage: <# iterations> <matrix order> [tile size]";
}
iterations  = std::atoi(argv[1]);
if (iterations < 1) {
throw "ERROR: iterations must be >= 1";
}
order = std::atoi(argv[2]);
if (order <= 0) {
throw "ERROR: Matrix Order must be greater than 0";
} else if (order > prk::get_max_matrix_size()) {
throw "ERROR: matrix dimension too large - overflow risk";
}
tile_size = (argc>3) ? std::atoi(argv[3]) : order;
if (tile_size <= 0) tile_size = order;
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations  = " << iterations << std::endl;
std::cout << "Matrix order          = " << order << std::endl;
#ifdef __GNUC__
std::cout << "Tile size             = " << "automatic (GCC)" << std::endl;
#else
std::cout << "Tile size             = " << tile_size << std::endl;
#endif
double trans_time{0};
size_t bytes = order*order*sizeof(double);
double * RESTRICT A = (double *)acc_malloc(bytes);
double * RESTRICT B = (double *)acc_malloc(bytes);
{
#pragma acc parallel loop deviceptr(A,B)
for (int i=0;i<order; i++) {
for (int j=0;j<order;j++) {
A[i*order+j] = static_cast<double>(i*order+j);
B[i*order+j] = 0.0;
}
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) trans_time = prk::wtime();
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
trans_time = prk::wtime() - trans_time;
}
const auto addit = (iterations+1.) * (iterations/2.);
auto abserr = 0.0;
#pragma acc parallel loop reduction( +:abserr ) deviceptr(B)
for (int j=0; j<order; j++) {
for (int i=0; i<order; i++) {
const size_t ij = i*order+j;
const size_t ji = j*order+i;
const double reference = static_cast<double>(ij)*(1.+iterations)+addit;
abserr += prk::abs(B[ji] - reference);
}
}
acc_free(A);
acc_free(B);
#ifdef VERBOSE
std::cout << "Sum of absolute differences: " << abserr << std::endl;
#endif
const auto epsilon = 1.0e-8;
if (abserr < epsilon) {
std::cout << "Solution validates" << std::endl;
auto avgtime = trans_time/iterations;
std::cout << "Rate (MB/s): " << 1.0e-6 * (2L*bytes)/avgtime
<< " Avg time (s): " << avgtime << std::endl;
} else {
std::cout << "ERROR: Aggregate squared error " << abserr
<< " exceeds threshold " << epsilon << std::endl;
return 1;
}
return 0;
}
