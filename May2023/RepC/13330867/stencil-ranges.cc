#include "prk_util.h"
#include "prk_ranges.h"
#include "stencil_ranges.hpp"
void nothing(const int n, const int t, prk::vector<double> & in, prk::vector<double> & out)
{
std::cout << "You are trying to use a stencil that does not exist.\n";
std::cout << "Please generate the new stencil using the code generator\n";
std::cout << "and add it to the case-switch in the driver." << std::endl;
if (n==0 || t==0) std::cout << in.size() << out.size() << std::endl;
std::abort();
}
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/ranges Stencil execution on 2D grid" << std::endl;
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
prk::vector<double> in(n*n);
prk::vector<double> out(n*n);
auto v = ranges::views::cartesian_product(ranges::views::iota(0, n),ranges::views::iota(0, n));
for (auto ij : v) {
auto [i, j] = ij;
in[i*n+j] = static_cast<double>(i+j);
out[i*n+j] = 0.0;
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk::wtime();
stencil(n, tile_size, in, out);
for (auto ij : v) {
auto [i, j] = ij;
in[i*n+j] += 1.0;
}
}
stencil_time = prk::wtime() - stencil_time;
size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
double norm = 0.0;
auto inside = prk::range(radius,n-radius);
for (auto i : inside) {
for (auto j : inside) {
norm += prk::abs(out[i*n+j]);
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
