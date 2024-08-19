#pragma once

#include <cstdio> 

#include "status.hxx" 
#include "data_view.hxx" 
#include "inline_math.hxx" 
#include "recorded_warnings.hxx" 

namespace dense_solver {

template <typename real_t>
inline real_t Lorentzian(real_t const re, real_t const im) { return -im/(pow2(im) + pow2(re)); }

template <typename real_t>
inline void display_spectrum(real_t const eigvals[], int const nB, char const *x_axis
, double const u=1, char const *_u="", char const *matrix_name="", int const mB=32) {
if (nB < 2) return;
std::printf("%s%s", x_axis, matrix_name);
for (int iB = 0; iB < std::min(nB - 2, mB - 2); ++iB) {
std::printf(" %g", eigvals[iB]*u);
} 
if (nB > mB) std::printf(" ..."); 
std::printf(" %g %g %s\n", eigvals[nB - 2]*u, eigvals[nB - 1]*u, _u); 
} 

template <typename complex_t>
status_t solve(
view3D<complex_t> & HSm 
, char const *x_axis
, int const echo=0 
, int const nbands=0 
, double *eigenenergies=nullptr 
); 

status_t all_tests(int const echo=0); 

} 
