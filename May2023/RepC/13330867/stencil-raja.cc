#include "prk_util.h"
#include "prk_raja.h"
#include "stencil_rajaview.hpp"
void nothing(const int n, const int t, matrix & in, matrix & out)
{
std::cout << "You are trying to use a stencil that does not exist.\n";
std::cout << "Please generate the new stencil using the code generator\n";
std::cout << "and add it to the case-switch in the driver." << std::endl;
std::abort();
}
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/RAJA Stencil execution on 2D grid" << std::endl;
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
double stencil_time{0};
double * RESTRICT imem = new double[n*n];
double * RESTRICT omem = new double[n*n];
RAJA::View<double, RAJA::Layout<2>> in(imem, n, n);
RAJA::View<double, RAJA::Layout<2>> out(omem, n, n);
using regular_policy = RAJA::KernelPolicy< RAJA::statement::For<0, thread_exec,
RAJA::statement::For<1, RAJA::simd_exec,
RAJA::statement::Lambda<0> > > >;
using permute_policy = RAJA::KernelPolicy< RAJA::statement::For<1, thread_exec,
RAJA::statement::For<0, RAJA::simd_exec,
RAJA::statement::Lambda<0> > > >;
RAJA::RangeSegment range(0, n);
auto grid = RAJA::make_tuple(range, range);
RAJA::kernel<regular_policy>(grid, [=](int i, int j) {
in(i,j)  = static_cast<double>(i+j);
out(i,j) = 0.0;
});
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk::wtime();
stencil(n, tile_size, in, out);
RAJA::kernel<regular_policy>(grid, [=](int i, int j) {
in(i,j) += 1.0;
});
}
stencil_time = prk::wtime() - stencil_time;
size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
RAJA::RangeSegment inside(radius,n-radius);
RAJA::ReduceSum<RAJA::seq_reduce, double> reduced_norm(0.0);
RAJA::forall<RAJA::seq_exec>(inside, [&](RAJA::Index_type i) {
RAJA::forall<RAJA::seq_exec>(inside, [&](RAJA::Index_type j) {
reduced_norm += prk::abs(out(i,j));
});
});
double norm = reduced_norm / active_points;
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
