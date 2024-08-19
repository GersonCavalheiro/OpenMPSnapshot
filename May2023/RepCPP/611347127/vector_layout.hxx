#pragma once

#include <cstdio> 
#include <cassert> 

#include "status.hxx" 

template <typename real_t>
class VecLayout {
public:

void axpby(real_t y[], real_t const x[], real_t const *a=nullptr, real_t const *b=nullptr) const {
auto const m = stride(); 
for (int j = 0; j < nrhs_; ++j) {
auto const sx = (a? a[j] :0), sy = (b? b[j] :0);
for (int i = 0; i < ndof_; ++i) {
auto const ij = i*m + j;
y[ij] = sx*x[ij] + sy*y[ij];
} 
} 
} 

void inner(double d[], real_t const x[], real_t const *y=nullptr) const {
auto const _y = y? y : x;
auto const m = stride(); 
for (int j = 0; j < nrhs_; ++j) {
double dj = 0;
for (int i = 0; i < ndof_; ++i) {
dj += x[i*m + j]*_y[i*m + j]; 
} 
d[j] = dj;
} 
} 

VecLayout(size_t ndof, size_t nrhs) : ndof_(ndof), nrhs_(nrhs) {} 

inline size_t nrhs() const { return nrhs_; }
inline size_t ndof() const { return ndof_; }
inline size_t stride() const { return (nrhs_)*r1c2_; } 
inline bool is_complex() const { return r1c2_ - 1; }

private:
size_t ndof_{0}; 
size_t nrhs_{0}; 
int r1c2_{1}; 
}; 



namespace vector_layout {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_construction_and_destruction(int const echo=9) {
if (echo > 0) std::printf("# %s\n", __func__);
typedef double real_t;
VecLayout<real_t> layout(8, 8);
return 0;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("# %s\n", __func__);
status_t stat(0);
stat += test_construction_and_destruction(echo);
return stat;
} 

#endif 

} 
