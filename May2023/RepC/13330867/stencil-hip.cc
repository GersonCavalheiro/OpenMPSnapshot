#include "prk_util.h"
#include "prk_hip.h"
#include "stencil_cuda.hpp"
__global__ void nothing(const int n, const prk_float * in, prk_float * out)
{
}
__global__ void add(const int n, prk_float * in)
{
auto i = blockIdx.x * blockDim.x + threadIdx.x;
auto j = blockIdx.y * blockDim.y + threadIdx.y;
if ((i<n) && (j<n)) {
in[i*n+j] += (prk_float)1;
}
}
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/HIP Stencil execution on 2D grid" << std::endl;
prk::HIP::info info;
info.print();
int iterations, n, radius, tile_size;
bool star = true;
try {
if (argc < 3) {
throw "Usage: <# iterations> <array dimension> [<tile_size> <star/grid> <radius>]";
}
iterations  = std::atoi(argv[1]);
if (iterations < 1) {
throw "ERROR: iterations must be >= 1";
}
n  = std::atoi(argv[2]);
if (n < 1) {
throw "ERROR: grid dimension must be positive";
} else if (n > prk::get_max_matrix_size()) {
throw "ERROR: grid dimension too large - overflow risk";
}
tile_size = 32;
if (argc > 3) {
tile_size = std::atoi(argv[3]);
if (tile_size <= 0) tile_size = n;
if (tile_size > n) tile_size = n;
if (tile_size > 32) {
std::cout << "Warning: tile_size > 32 may lead to incorrect results (observed for HIP 9.0 on GV100).\n";
}
}
if (argc > 4) {
auto stencil = std::string(argv[4]);
auto grid = std::string("grid");
star = (stencil == grid) ? false : true;
}
radius = 2;
if (argc > 5) {
radius = std::atoi(argv[5]);
}
if ( (radius < 1) || (2*radius+1 > n) ) {
throw "ERROR: Stencil radius negative or too large";
}
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations = " << iterations << std::endl;
std::cout << "Grid size            = " << n << std::endl;
std::cout << "Tile size            = " << tile_size << std::endl;
std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
std::cout << "Radius of stencil    = " << radius << std::endl;
auto stencil = nothing;
if (star) {
switch (radius) {
case 1: stencil = star1; break;
case 2: stencil = star2; break;
case 3: stencil = star3; break;
case 4: stencil = star4; break;
case 5: stencil = star5; break;
}
} else {
switch (radius) {
case 1: stencil = grid1; break;
case 2: stencil = grid2; break;
case 3: stencil = grid3; break;
case 4: stencil = grid4; break;
case 5: stencil = grid5; break;
}
}
dim3 dimGrid(prk::divceil(n,tile_size),prk::divceil(n,tile_size),1);
dim3 dimBlock(tile_size, tile_size, 1);
info.checkDims(dimBlock, dimGrid);
auto stencil_time = 0.0;
const size_t nelems = (size_t)n * (size_t)n;
const size_t bytes = nelems * sizeof(prk_float);
prk_float * h_in;
prk_float * h_out;
prk::HIP::check( hipHostMalloc((void**)&h_in, bytes) );
prk::HIP::check( hipHostMalloc((void**)&h_out, bytes) );
for (int i=0; i<n; i++) {
for (int j=0; j<n; j++) {
h_in[i*n+j]  = static_cast<prk_float>(i+j);
h_out[i*n+j] = static_cast<prk_float>(0);
}
}
prk_float * d_in;
prk_float * d_out;
prk::HIP::check( hipMalloc((void**)&d_in, bytes) );
prk::HIP::check( hipMalloc((void**)&d_out, bytes) );
prk::HIP::check( hipMemcpy(d_in, &(h_in[0]), bytes, hipMemcpyHostToDevice) );
prk::HIP::check( hipMemcpy(d_out, &(h_out[0]), bytes, hipMemcpyHostToDevice) );
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk::wtime();
hipLaunchKernelGGL(stencil, dim3(dimGrid), dim3(dimBlock), 0, 0, n, d_in, d_out);
hipLaunchKernelGGL(add, dim3(dimGrid), dim3(dimBlock), 0, 0, n, d_in);
prk::HIP::check( hipDeviceSynchronize() );
}
stencil_time = prk::wtime() - stencil_time;
prk::HIP::check( hipMemcpy(&(h_out[0]), d_out, bytes, hipMemcpyDeviceToHost) );
#ifdef VERBOSE
prk::HIP::check( hipMemcpy(&(h_in[0]), d_in, bytes, hipMemcpyDeviceToHost) );
#endif
prk::HIP::check( hipFree(d_out) );
prk::HIP::check( hipFree(d_in) );
size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
double norm = 0.0;
for (int i=radius; i<n-radius; i++) {
for (int j=radius; j<n-radius; j++) {
norm += prk::abs(h_out[i*n+j]);
}
}
norm /= active_points;
const double epsilon = 1.0e-8;
double reference_norm = 2.*(iterations+1.);
if (prk::abs(norm-reference_norm) > epsilon) {
std::cout << "ERROR: L1 norm = " << norm
<< " Reference L1 norm = " << reference_norm << std::endl;
return 1;
} else {
std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
std::cout << "L1 norm = " << norm
<< " Reference L1 norm = " << reference_norm << std::endl;
#endif
const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
auto avgtime = stencil_time/iterations;
std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
<< " Avg time (s): " << avgtime << std::endl;
}
return 0;
}
