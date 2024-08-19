#pragma once

#include <cstdio> 
#include <algorithm> 
#include <cmath> 

#include "status.hxx" 
#include "data_view.hxx" 
#include "inline_math.hxx" 
#include "print_tools.hxx" 

#include "control.hxx" 

namespace brillouin_zone {

int constexpr WEIGHT = 3; 

inline int get_kpoint_mesh( 
view2D<double> & mesh 
, unsigned const nv[3] 
, bool const complex_phase_factors=true
, int const echo=0 
) {
unsigned n[3];
double shift[3];
for (int d = 0; d < 3; ++d) {
n[d] = std::max(1u, nv[d]);
shift[d] = n[d] - 1.;
if (!complex_phase_factors) {
n[d] = std::min(n[d], 2u); 
shift[d] = 0;
} 
} 

int constexpr COUNT=0, WRITE=1;

size_t const nfull = n[2]*n[1]*n[0];
view4D<double> full(n[2], n[1], n[0], 4); 
double const denom[] = {.5/n[0], .5/n[1], .5/n[2]};
double xyzw[] = {0, 0, 0, 1};
for (int iz = 0; iz < n[2]; ++iz) {  xyzw[2] = (2*iz - shift[2])*denom[2];
for (int iy = 0; iy < n[1]; ++iy) {  xyzw[1] = (2*iy - shift[1])*denom[1];
for (int ix = 0; ix < n[0]; ++ix) {  xyzw[0] = (2*ix - shift[0])*denom[0];
set(full(iz, iy, ix), 4, xyzw);
if (echo > 18) std::printf("# k-point mesh entry %9.6f %9.6f %9.6f weight= %g\n", xyzw[0],xyzw[1],xyzw[2], xyzw[3]);
}}} 


for (int iz = 0; iz < n[2]; ++iz) {
for (int iy = 0; iy < n[1]; ++iy) {
for (int ix = 0; ix < n[0]; ++ix) {
double xyz[3];
set(xyz, 3, full(iz,iy,ix), -1.0); 
int jxyz[3];
for (int d = 0; d < 3; ++d) {
jxyz[d] = int(std::round(n[d]*xyz[d] + 0.5*shift[d]));
if (jxyz[d] < 0) jxyz[d] += n[d];
} 
int const jx = jxyz[0], jy = jxyz[1], jz = jxyz[2];
if (echo > 16) {
std::printf("# symmetry #%i maps k-point", 0);
printf_vector(" %9.6f", full(iz,iy,ix), 3, " to");
printf_vector(" %9.6f", full(jz,jy,jx), 3);
} 

auto const w8 = full(jz,jy,jx,WEIGHT);
if (w8 > 0) {
double diff2{0};
for (int d = 0; d < 3; ++d) {
diff2 += pow2(xyz[d] - full(jz,jy,jx,d));
} 
if (diff2 < 1e-12) {
full(iz,iy,ix,WEIGHT) += w8;
full(jz,jy,jx,WEIGHT) -= w8;
if (echo > 14) {
std::printf("# transfer k-point weight %g from", w8);
printf_vector(" %9.6f", full(jz,jy,jx), 3, " to");
printf_vector(" %9.6f", full(iz,iy,ix), 3);
} 
} 
} 
}}} 

double const wfull = 1./nfull;
double const weighted[] = {1, 1, 1, wfull};
int nmesh{0};
for (int i01 = COUNT; i01 <= WRITE; i01 += (WRITE - COUNT)) {
double w8sum{0};
int imesh{0};
for (int iz = 0; iz < n[2]; ++iz) {
for (int iy = 0; iy < n[1]; ++iy) {
for (int ix = 0; ix < n[0]; ++ix) {
auto const w8 = full(iz,iy,ix,WEIGHT);
if (w8 > 0) {
if (WRITE == i01) product(mesh[imesh], 4, full(iz,iy,ix), weighted); 
w8sum += w8;
++imesh;
} 
}}} 
if (WRITE == i01) {
assert(nmesh == imesh && "1st and 2nd time counting did not agree");
} else {
nmesh = imesh;
if (echo > 8) std::printf("# %d k-points have positive weight, weight sum = 1 + %.1e\n", nmesh, w8sum*wfull - 1);
mesh = view2D<double>(nmesh, 4); 
} 
} 

if (echo > 3) std::printf("# k-point mesh with %d x %d x %d has %d points\n", n[0],n[1],n[2], nmesh);
return nmesh;
} 


inline int get_kpoint_mesh(
view2D<double> & mesh
, unsigned const n 
, bool const complex_phase_factors=true
, int const echo=0 
) {
unsigned const nv[] = {n, n, n};
return get_kpoint_mesh(mesh, nv, complex_phase_factors, echo);
} 


inline int get_kpoint_mesh(
view2D<double> & mesh
) {
unsigned nv[3];
auto const iso = control::get("hamiltonian.kmesh", 1.); 
nv[0]          = control::get("hamiltonian.kmesh.x", iso);
nv[1]          = control::get("hamiltonian.kmesh.y", iso);
nv[2]          = control::get("hamiltonian.kmesh.z", iso);
int const echo = control::get("hamiltonian.kmesh.echo", 0.);
int const cmpl = control::get("hamiltonian.kmesh.complex", 1.);
return get_kpoint_mesh(mesh, nv, (0 != cmpl), echo);
} 


inline bool needs_complex(double const kvec[3]) {
return !(is_integer(2*kvec[0])
&& is_integer(2*kvec[1])
&& is_integer(2*kvec[2]));
} 


inline bool needs_complex(
view2D<double> const & mesh
, int const nmesh
) {
for (int ik = 0; ik < nmesh; ++ik) {
if (needs_complex(mesh[ik])) return true;
} 
return false;
} 


inline char get_special_kpoint(double v[3], char const *const keyword, char const *const def="?") {
v[0] = 0; v[1] = 0; v[2] = 0; 
char const cin = *control::get(keyword, def);
switch (cin | 32) { 
case 'g': case '0':                           return 'G';
case 'x': case '1':               v[0] = 0.5; return 'X';
case 'm':             v[1] = 0.5; v[0] = 0.5; return 'M';
case 'r': v[2] = 0.5; v[1] = 0.5; v[0] = 0.5; return 'R';
default:
{ 
char extended_keyword[64];
for (int d = 0; d < 3; ++d) {
std::snprintf(extended_keyword, 64, "%s.%c", keyword, 'x' + d);
v[d] = control::get(extended_keyword, 0.0);
} 
} 
} 
return cin;
} 


inline int get_kpoint_path( 
view2D<double> & path 
, int const echo=0 
) {
double from[3], to[3];
auto const c_from = get_special_kpoint(from, "hamiltonian.kpath.from", "Gamma");
auto const c_to   = get_special_kpoint(to,   "hamiltonian.kpath.to",   "X");
auto const dk = control::get("hamiltonian.kpath.spacing", 0.05);
auto const length = std::sqrt(pow2(to[0] - from[0]) + pow2(to[1] - from[1]) + pow2(to[2] - from[2]));
int const nk = std::ceil(length/std::max(dk, 1e-9));
int const npath = nk + 1;
path = view2D<double>(npath, 4, 0.0); 
double const by_nk = (nk > 0) ? 1./nk : 0;
for (int ik = 0; ik <= nk; ++ik) {
for (int d = 0; d < 3; ++d) {
path(ik,d) = from[d] + (to[d] - from[d])*(ik*by_nk);
} 
path(ik,3) = length*(ik*by_nk); 
} 
if (echo > 0) std::printf("# k-point path from %g %g %g (%c) to %g %g %g (%c) in %d steps, dk= %g\n",
from[0], from[1], from[2], c_from, to[0], to[1], to[2], c_to, nk, length*by_nk);
return npath;
} 









#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_mesh(int const echo=0, unsigned const n=8) {
view2D<double> mesh;
auto const nm = get_kpoint_mesh(mesh, n, echo);
if (echo > 5) std::printf("# %s(n= %d) returns %d points\n", __func__, n, nm);
return (nm < 1); 
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
for (int n = 0; n < 9; ++n) {
stat += test_mesh(echo, n);
} 
return stat;
} 

#endif 

} 
