#include "prk_util.h"
#include "p2p-kernel.h"
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/ORNL-ACC HYPERPLANE pipeline execution on 2D grid" << std::endl;
int iterations;
int n, nc, nb;
try {
if (argc < 3) {
throw " <# iterations> <array dimension> [<chunk dimension>]";
}
iterations  = std::atoi(argv[1]);
if (iterations < 0) {
throw "ERROR: iterations must be >= 1";
}
n = std::atoi(argv[2]);
if (n < 1) {
throw "ERROR: grid dimensions must be positive";
} else if ( n > prk::get_max_matrix_size() ) {
throw "ERROR: grid dimension too large - overflow risk";
}
nc = (argc > 3) ? std::atoi(argv[3]) : 1;
nc = std::max(1,nc);
nc = std::min(n,nc);
nb = (n-1)/nc;
if ((n-1)%nc) nb++;
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations = " << iterations << std::endl;
std::cout << "Grid sizes           = " << n << ", " << n << std::endl;
std::cout << "Grid chunk sizes     = " << nc << std::endl;
double pipeline_time{0};
double * grid = new double[n*n];
for (int i=0; i<n; i++) {
for (int j=0; j<n; j++) {
grid[i*n+j] = 0.0;
}
}
for (int j=0; j<n; j++) {
grid[0*n+j] = static_cast<double>(j);
}
for (int i=0; i<n; i++) {
grid[i*n+0] = static_cast<double>(i);
}
#pragma acc data pcopy(grid[0:n*n])
{
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) pipeline_time = prk::wtime();
if (nc==1) {
for (int i=2; i<=2*n-2; i++) {
#pragma acc parallel loop independent
for (int j=std::max(2,i-n+2); j<=std::min(i,n); j++) {
const int x = i-j+1;
const int y = j-1;
grid[x*n+y] = grid[(x-1)*n+y] + grid[x*n+(y-1)] - grid[(x-1)*n+(y-1)];
}
}
} else {
for (int i=2; i<=2*(nb+1)-2; i++) {
#pragma acc parallel loop gang
for (int j=std::max(2,i-(nb+1)+2); j<=std::min(i,nb+1); j++) {
const int ib = nc*(i-j)+1;
const int jb = nc*(j-2)+1;
#pragma acc loop vector
for (int i=ib; i<std::min(n,ib+nc); i++) {
for (int j=jb; j<std::min(n,jb+nc); j++) {
grid[i*n+j] = grid[(i-1)*n+j] + grid[i*n+(j-1)] - grid[(i-1)*n+(j-1)];
}
}
}
}
}
#pragma acc kernels
{
grid[0*n+0] = -grid[(n-1)*n+(n-1)];
}
}
pipeline_time = prk::wtime() - pipeline_time;
}
const double epsilon = 1.e-8;
auto corner_val = ((iterations+1.)*(2.*n-2.));
if ( (prk::abs(grid[(n-1)*n+(n-1)] - corner_val)/corner_val) > epsilon) {
std::cout << "ERROR: checksum " << grid[(n-1)*n+(n-1)]
<< " does not match verification value " << corner_val << std::endl;
return 1;
}
#ifdef VERBOSE
std::cout << "Solution validates; verification value = " << corner_val << std::endl;
#else
std::cout << "Solution validates" << std::endl;
#endif
auto avgtime = pipeline_time/iterations;
std::cout << "Rate (MFlops/s): "
<< 2.0e-6 * ( (n-1.)*(n-1.) )/avgtime
<< " Avg time (s): " << avgtime << std::endl;
return 0;
}
