#pragma once

#include <cstdio> 

#include "display_units.h" 
#include "complex_tools.hxx" 

#ifdef DEBUG
#define here \
if (echo > 5) { \
std::printf("\n# here: %s %s:%i\n\n", __func__, __FILE__, __LINE__); \
std::fflush(stdout); \
}
#else  
#define here ;
#endif 

template <typename real_t>
inline int dump_to_file(
char const *filename
, size_t const N 
, real_t const y_data[] 
, double *x_axis=nullptr 
, size_t const Stride=1 
, size_t const M=1 
, char const *title=nullptr 
, int const echo=0 
) {
auto *const f = std::fopen(filename, "w");
if (nullptr == f) {
if (echo > 1) std::printf("# %s Error opening file %s!\n", __func__, filename);
return 1;
} 

std::fprintf(f, "#%s %s\n", (is_complex<real_t>())?"complex":"", title); 

for (int i = 0; i < N; i++) {
if (nullptr != x_axis) {
std::fprintf(f, "%g ", x_axis[i]);
} else {
std::fprintf(f, "%d ", i);
} 
for (int j = 0; j < M; ++j) {
auto const y = y_data[i*Stride + j];
if (is_complex<real_t>()) {
std::fprintf(f, "  %g %g", std::real(y), std::imag(y));
} else {
std::fprintf(f, " %g", std::real(y));
} 
} 
std::fprintf(f, "\n"); 
} 
std::fprintf(f, "\n"); 

std::fclose(f);
if (echo > 3) std::printf("# file %s written with %lu x %lu (of %lu) data entries.\n", filename, N, M, Stride);
return 0;
} 

namespace debug_output {

template <typename real_t>
inline status_t write_array_to_file(
char const *filename 
, real_t const array[]  
, int const nx, int const ny, int const nz 
, int const echo=0 
, char const *arrayname="" 
) {
char title[128]; std::snprintf(title, 128, "%i x %i x %i  %s", nz, ny, nx, arrayname);
auto const size = size_t(nz) * size_t(ny) * size_t(nx);
return dump_to_file(filename, size, array, nullptr, 1, 1, title, echo);
} 

} 
