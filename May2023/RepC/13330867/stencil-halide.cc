#include "prk_util.h"
#include "Halide.h"
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/Halide Stencil execution on 2D grid" << std::endl;
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
} else if (n > std::floor(std::sqrt(INT_MAX))) {
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
const Halide::Target target = Halide::get_jit_target_from_environment();
double stencil_time(0);
Halide::Buffer<double> in(n,n);
Halide::Buffer<double> out(n,n);
Halide::Var x("x");
Halide::Var y("y");
Halide::Expr c1(0.25);
Halide::Expr c2(0.125);
Halide::Func stencil;
stencil(x,y) = c1 * ( in(x+1,y) + in(x-1,y) + in(x,y+1) + in(x,y+1) )
+ c2 * ( in(x+2,y) + in(x-2,y) + in(x,y+2) + in(x,y+2) );
{
for (auto i=0; i<n; ++i) {
for (auto j=0; j<n; ++j) {
in(i,j)  = 1.0*(i+j);
out(i,j) = 0.0;
}
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk::wtime();
out = stencil.realize(n,n);
for (int i=0; i<n; ++i) {
for (int j=0; j<n; ++j) {
in(i,j) += 1.0;
}
}
}
stencil_time = prk::wtime() - stencil_time;
}
size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
double norm = 0.0;
for (auto i=radius; i<n-radius; i++) {
for (auto j=radius; j<n-radius; j++) {
norm += std::fabs(out(i,j));
}
}
norm /= active_points;
const double epsilon = 1.0e-8;
double reference_norm = 2.*(iterations+1.);
if (std::fabs(norm-reference_norm) > epsilon) {
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
