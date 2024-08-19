#pragma once

#include <cstdio> 
#include <vector> 
#include <cmath> 
#include <algorithm> 

#include "status.hxx" 
#include "radial_grid.h" 
#include "energy_level.hxx" 
#include "display_units.h" 
#include "inline_math.hxx" 
#include "simple_math.hxx" 
#include "linear_algebra.hxx" 
#include "print_tools.hxx" 
#include "recorded_warnings.hxx" 
#include "constants.hxx" 
#include "data_view.hxx" 


namespace pseudo_tools {

inline status_t pseudize_function(
double fun[] 
, double const rg[] 
, int const irc 
, int const nmax=4 
, int const ell=0 
, double *coeff=nullptr 
, int const echo=0 
) {

double Amat[4][4], bvec[4] = {0,0,0,0};
set(Amat[0], 4*4, 0.0);
int const nm = std::min(std::max(1, nmax), 4);
for (int i4 = 0; i4 < nm; ++i4) {
int const ir = irc + i4 - nm/2; 
double const r = rg[ir];
double rl = intpow(r, ell);
for (int j4 = 0; j4 < nm; ++j4) {
Amat[j4][i4] = rl;
rl *= pow2(r);
} 
bvec[i4] = fun[ir];
} 
#ifdef DEVEL
if (echo > 7) {
std::printf("\n");
for (int i4 = 0; i4 < nm; ++i4) {
std::printf("# %s Amat i=%i ", __func__, i4);
for (int j4 = 0; j4 < nm; ++j4) {
std::printf(" %16.9f", Amat[j4][i4]);
} 
std::printf(" bvec %16.9f\n", bvec[i4]);
} 
} 
#endif 
auto const info = linear_algebra::linear_solve(nm, Amat[0], 4, bvec, 4, 1);
double* x = bvec; 
#ifdef DEVEL
if (echo > 7) {
std::printf("# %s xvec     ", __func__);
printf_vector(" %16.9f", x, nm);
} 
#endif 
set(x + nm, 4 - nm, 0.0); 
for (int ir = 0; ir < irc; ++ir) {
double const r = rg[ir];
double const rr = pow2(r);
double const rl = intpow(r, ell);
fun[ir] = rl*(x[0] + rr*(x[1] + rr*(x[2] + rr*x[3])));
} 
if (nullptr != coeff) set(coeff, nm, x); 
return info;
} 


inline double pseudize_spherical_density(
double smooth_density[]
, double const true_density[]
, radial_grid_t const rg[TRU_AND_SMT] 
, int const ir_cut[TRU_AND_SMT] 
, char const *quantity="core"
, char const *label="" 
, int const echo=0 
) {
int const nrs = rg[SMT].n;
int const nr_diff = rg[TRU].n - nrs;
set(smooth_density, nrs, true_density + nr_diff); 

auto const stat = pseudize_function(smooth_density, rg[SMT].r, ir_cut[SMT], 3); 
if (stat) warn("%s Matching procedure for the smooth %s density failed! info= %d", label, quantity, int(stat));
#ifdef DEVEL
if (echo > 8) { 
std::printf("\n## %s radius, {smooth, true} for %s density:\n", label, quantity);
for (int ir = 0; ir < nrs; ir += 2) { 
std::printf("%g %g %g\n", rg[SMT].r[ir], smooth_density[ir], true_density[ir + nr_diff]);
} 
std::printf("\n\n");
} 
#endif 
auto const tru_charge = dot_product(rg[TRU].n, rg[TRU].r2dr, true_density);
auto const smt_charge = dot_product(rg[SMT].n, rg[SMT].r2dr, smooth_density);
double const charge_deficit = tru_charge - smt_charge;
if (echo > 2) std::printf("# %s true and smooth %s density have %g and %g electrons, respectively\n",
label,             quantity,       tru_charge, smt_charge);
return charge_deficit;
} 


template <typename real_t>
status_t Lagrange_derivatives(
unsigned const n 
, real_t const y[] 
, double const x[] 
, double const x0 
, double *value_x0 
, double *deriv1st 
, double *deriv2nd 
) {

double d0{0}, d1{0}, d2{0};

for (int j = 0; j < n; ++j) {
double d0j{1}, d1j{0}, d2j{0};
for (int k = 0; k < n; ++k) {
if (k != j) {
if (std::abs(x[j] - x[k]) < 1e-9*std::max(std::abs(x[j]), std::abs(x[j]))) return 1; 
double d1l{1}, d2l{0};
for (int l = 0; l < n; ++l) {
if (l != j && l != k) {

double d2m{1};
for (int m = 0; m < n; ++m) {
if (m != j && m != k && m != l) {
d2m *= (x0 - x[m])/(x[j] - x[m]);
} 
} 

d1l *= (x0 - x[l])/(x[j] - x[l]);
d2l += d2m/(x[j] - x[l]);
} 
} 

d0j *= (x0 - x[k])/(x[j] - x[k]);
d1j += d1l/(x[j] - x[k]);
d2j += d2l/(x[j] - x[k]);
} 
} 
d0 += d0j * y[j];
d1 += d1j * y[j];
d2 += d2j * y[j];
} 

if(value_x0) *value_x0 = d0;
if(deriv1st) *deriv1st = d1;
if(deriv2nd) *deriv2nd = d2;

return 0; 
} 


template <int rpow>
status_t pseudize_local_potential(
double V_smt[]  
, double const V_tru[] 
, radial_grid_t const rg[TRU_AND_SMT] 
, int const ir_cut[TRU_AND_SMT] 
, int const n_Lagrange_points=0 
, char const *label="" 
, int const echo=0 
, double const df=1 
) {
int const nr_diff = ir_cut[TRU] - ir_cut[SMT];
double const r_cut = rg[TRU].r[ir_cut[TRU]];

assert(rg[TRU].r[nr_diff] == rg[SMT].r[0]);
assert(r_cut == rg[SMT].r[ir_cut[SMT]]);

if (echo > 2) std::printf("\n# %s %s\n", label, __func__);
status_t stat(0);
if (n_Lagrange_points < 1) {
if (echo > 2) std::printf("\n# %s construct initial smooth spherical potential as parabola\n", label);
set(V_smt, rg[SMT].n, V_tru + nr_diff); 
stat = pseudize_function(V_smt, rg[SMT].r, ir_cut[SMT], 2, rpow); 
if (echo > 5) std::printf("# %s match local potential to parabola at R_cut = %g %s, V_tru(R_cut) = %g %s\n",
label, r_cut*Ang, _Ang, V_tru[ir_cut[TRU]]*(rpow ? 1./r_cut : 1)*df*eV, _eV);
} else {
if (echo > 2) std::printf("\n# %s construct initial smooth spherical potential with sinc\n", label);
assert(rpow == rpow*rpow); 
int const ir0 = ir_cut[TRU];
double const x0 = rg[TRU].r[ir0];
double xi[32], yi[32];
yi[0] = V_tru[ir0]*(rpow ? 1 : x0); 
xi[0] = rg[TRU].r[ir0] - x0;  
for (int order = 1; order < 16; ++order) {
{
int const ir = ir0 + order; assert(ir < rg[TRU].n);
int const i = 2*order - 1;
yi[i] = V_tru[ir]*(rpow ? 1 : rg[TRU].r[ir]);
xi[i] = rg[TRU].r[ir] - x0;
}
{
int const ir = ir0 - order; assert(ir >= 0);
int const i = 2*order;
yi[i] = V_tru[ir]*(rpow ? 1 : rg[TRU].r[ir]);
xi[i] = rg[TRU].r[ir] - x0;
}
} 
#ifdef DEVEL
if (echo > 22) {
std::printf("\n## Lagrange-N, reference-value, Lagrange(rcut), dLagrange/dr, d^2Lagrange/dr^2, status:\n");
for (int order = 1; order < 16; ++order) {
unsigned const Lagrange_order = 2*order + 1;
{
double d0{0}, d1{0}, d2{0};
auto const stat = Lagrange_derivatives(Lagrange_order, yi, xi, 0, &d0, &d1, &d2);
std::printf("%d %.15f %.15f %.15f %.15f %i\n", Lagrange_order, V_tru[ir0], d0, d1, d2, int(stat));
}
} 
} 

if ((1 == rpow) && (echo > 27)) {
for (int order = 1; order < 16; ++order) {
unsigned const Lagrange_order = 2*order + 1;
std::printf("\n\n## r, r*V_true(r), Lagrange_fit(N=%d), dLagrange/dr, d^2Lagrange/dr^2, status:\n", Lagrange_order);
for (int ir = 0; ir < rg[TRU].n; ++ir) {
double d0{0}, d1{0}, d2{0};
auto const stat = Lagrange_derivatives(Lagrange_order, yi, xi, rg[TRU].r[ir] - x0, &d0, &d1, &d2);
std::printf("%g %g %g %g %g %i\n", rg[TRU].r[ir], V_tru[ir], d0, d1, d2, int(stat));
} 
std::printf("\n\n");
} 
} 
#endif 
unsigned const Lagrange_order = 1 + 2*std::min(std::max(1, n_Lagrange_points ), 15);
double d0{0}, d1{0}, d2{0};
stat = Lagrange_derivatives(Lagrange_order, yi, xi, 0, &d0, &d1, &d2);
if (echo > 7) std::printf("# %s use %d points, %g =value= %g derivative=%g second=%g status=%i\n",
label, Lagrange_order, yi[0], d0, d1, d2, int(stat));

if (d1 <= 0) warn("%s positive potential slope for sinc-fit expected but found %g", label, d1);
if (d2 >  0) warn("%s negative potential curvature for sinc-fit expected but found %g", label, d2);

double k_s{2.25*constants::pi/r_cut}, k_s_prev{0}; 
double V_s{0}, V_0{0};
int const max_iter = 999;
int iterations_needed{0};
for (int iter = max_iter; (std::abs(k_s - k_s_prev) > 1e-15) && (iter > 0); --iter) {
k_s_prev = k_s;
double const sin_kr = std::sin(k_s*r_cut);
double const cos_kr = std::cos(k_s*r_cut);
V_s = d2/(-k_s*k_s*sin_kr);
V_0 = d1 - V_s*k_s*cos_kr;
double const d0_new = 0.5*d0 + 0.5*(V_s*sin_kr + r_cut*V_0); 
if (echo > 27) std::printf("# %s iter=%i use V_s=%g k_s=%g V_0=%g d0=%g d0_new-d0=%g\n",
label, max_iter - iter, V_s, k_s, V_0, d0, d0 - d0_new);
k_s += 1e-2*(d0 - d0_new); 
++iterations_needed;
} 
if (iterations_needed >= max_iter) warn("%s sinc-fit did not converge in %d iterations!", label, iterations_needed);
if (k_s*r_cut <= 2*constants::pi || k_s*r_cut >= 2.5*constants::pi) {
warn("%s sinc-fit failed!, k_s*r_cut=%g not in (2pi, 2.5pi)", label, k_s*r_cut);
} 

if (echo > 5) std::printf("# %s match local potential to sinc function at R_cut = %g %s, V_tru(R_cut) = %g %s\n",
label, r_cut*Ang, _Ang, yi[0]/r_cut*df*eV, _eV);

if (echo > 3) std::printf("# %s smooth potential value of sinc-fit at origin is %g %s\n", label, (V_s*k_s + V_0)*df*eV, _eV);

for (int ir = 0; ir <= ir_cut[SMT]; ++ir) {
double const r = rg[SMT].r[ir];
V_smt[ir] = (V_s*std::sin(k_s*r) + r*V_0)*(rpow ? 1 : rg[SMT].rinv[ir]); 
} 
for (int ir = ir_cut[SMT]; ir < rg[SMT].n; ++ir) {
V_smt[ir] = V_tru[ir + nr_diff]; 
} 

} 
#ifdef DEVEL
if (echo > 11) {
std::printf("\n\n## %s: r in %s, r*V_tru(r), r*V_smt(r) in %s*%s:\n", __func__, _Ang, _Ang, _eV);
auto const factors = df*eV*Ang;
for (int ir = 0; ir < rg[SMT].n; ++ir) {
double const r = rg[SMT].r[ir], r_pow = rpow ? 1 : r;
std::printf("%g %g %g\n", r*Ang, r_pow*V_tru[ir + nr_diff]*factors, r_pow*V_smt[ir]*factors);
} 
std::printf("\n\n");
} 
#endif 
return stat;
} 


inline double perform_Gram_Schmidt( 
int const n 
, view3D<double> & LU_inv 
, view2D<double> const & A 
, char const *label 
, char const ell='?' 
, int const echo=0 
, char const op[]="UUUUUUUUUUUUUUU" 
) {

if (n < 1) return 0.0; 
double const det = simple_math::determinant(n, A.data(), A.stride());
if (echo > 4) std::printf("# %s determinant for %c-projectors is %g\n", label, ell, det);
if (std::abs(det) < 1e-24) {
warn("%s Determinant of <projectors|partial waves> for ell=%c is %.1e", label, ell, det);
return det;
}
view2D<double> L(n, n, 0.0), U(n, n, 0.0); 
for (int i = 0; i < n; ++i) {
U(i,i) = 1; 
set(L[i], n, A[i]); 
} 

for (int i = 0; i < n; ++i) {

if (op[i] & 32) { 
auto const diag = L(i,i);
auto const dinv = 1./diag; 
for (int k = 0; k < n; ++k) {
U(i,k) *= diag; 
L(k,i) *= dinv; 
} 
} else {
} 

for (int j = i + 1; j < n; ++j) {

if (op[j] & 1) { 
auto const c = L(i,j)/L(i,i); 
for (int k = 0; k < n; ++k) {
L(k,j) -= c*L(k,i); 
U(i,k) += c*U(j,k); 
} 
L(i,j) = 0; 
} 

else { 
auto const c = L(j,i)/L(j,j); 
for (int k = 0; k < n; ++k) {
L(k,i) -= c*L(k,j); 
U(j,k) += c*U(i,k); 
} 
L(j,i) = 0; 
} 

} 
} 

auto L_inv = LU_inv[0], U_inv = LU_inv[1];
double const det_U = simple_math::invert(n, U_inv.data(), U_inv.stride(), U.data(), n);
double const det_L = simple_math::invert(n, L_inv.data(), L_inv.stride(), L.data(), n);
if (echo > 91) std::printf("# %s: |U|= %g, |L|= %g\n", __func__, det_U, det_L);

#ifdef DEVEL
if (echo > 11) { 

for (int i = 0; i < n; ++i) {
std::printf("# %s  ell=%c  L(i,:) and U(i,:)  i=%2i ", label, ell, i);
printf_vector(" %11.6f", L[i], n, "\t\t");
printf_vector(" %11.6f", U[i], n);
} 
std::printf("\n");

view2D<double> A_LU(n, n);
gemm(A_LU, n, L, n, U);
for (int i = 0; i < n; ++i) {
std::printf("# %s  ell=%c A, L*U, diff i=%2i ", label, ell, i);
if (0) {
printf_vector(" %11.6f", A[i], n, "\t\t");
printf_vector(" %11.6f", A_LU[i], n, "\t\t");
} 
for (int j = 0; j < n; ++j) {
std::printf(" %11.6f", A_LU(i,j) - A(i,j));
} 
std::printf("\n");
} 

std::printf("\n# %s  ell=%c  |L|= %g and |U|= %g, product= %g, |A|= %g\n",
label, ell, det_L, det_U, det_L*det_U, det);
for (int i = 0; i < n; ++i) {
std::printf("# %s  ell=%c  L^-1 and U^-1  i=%2i ", label, ell, i);
printf_vector(" %11.6f", L_inv[i], n, "\t\t\t");
printf_vector(" %11.6f", U_inv[i], n);
} 
std::printf("\n");
} 
#endif 
return det;
} 


#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_pseudize_function(int const echo=0, int const nr=96) {
status_t stat(0);
std::vector<double> r(nr), pseudized_function(nr), original_function(nr);
for (int ir = 0; ir < nr; ++ir) {
r[ir] = 0.1*ir; 
original_function[ir] = std::cos(r[ir]); 
pseudized_function[ir] = original_function[ir]; 
} 
stat += pseudize_function(pseudized_function.data(), r.data(), nr/2);
if (echo > 7) { 
std::printf("\n## r, original_function(r), pseudized_function(r):\n");
for (int ir = 0; ir < nr; ++ir) {
std::printf("%g %g %g\n", r[ir], original_function[ir], pseudized_function[ir]);
} 
} 
return stat;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_pseudize_function(echo);
return stat;
} 

#endif 

} 
