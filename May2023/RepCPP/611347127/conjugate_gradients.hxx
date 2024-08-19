#pragma once

#include <cstdio> 
#include <complex> 
#include <vector> 
#include <cmath> 
#include <algorithm> 

#include "status.hxx" 
#include "complex_tools.hxx" 
#include "inline_math.hxx" 
#include "data_view.hxx" 
#include "display_units.h" 
#include "recorded_warnings.hxx" 
#include "control.hxx" 

#ifndef NO_UNIT_TESTS
#include "simple_math.hxx" 
#include "grid_operators.hxx" 
#endif

namespace conjugate_gradients {

template <typename complex_t> inline double tiny();
template <> inline double tiny<double>() { return 2.25e-308; }
template <> inline double tiny<float> () { return 1.18e-38; }

template <typename doublecomplex_t, typename complex_t>
doublecomplex_t inner_product(
size_t const ndof
, complex_t const bra[] 
, complex_t const ket[] 
, doublecomplex_t const factor
) { 
doublecomplex_t tmp{0};
for (size_t dof = 0; dof < ndof; ++dof) {
tmp += doublecomplex_t(conjugate(bra[dof])) * doublecomplex_t(ket[dof]);
} 
return tmp*factor; 
} 

template <typename complex_t>
void show_matrix(
complex_t const mat[]
, int const stride
, int const n
, int const m
, char const *name=nullptr
, complex_t const unit=1
, char const initchar='#'
, char const final_newline='\n'
) {
if (is_complex<complex_t>()) return;
std::printf("%c %s=%s%c", initchar, (n > 1)?"Matrix":"Vector", name, (n > 1)?'\n':' ');
for (int i = 0; i < n; ++i) {
if (n > 1) std::printf("#%4i ", i);
for (int j = 0; j < m; ++j) {
std::printf("%9.3f", mat[i*stride + j]*unit);
} 
std::printf("\n");
} 
if (final_newline) std::printf("%c", final_newline);
} 



template <typename complex_t>
double submatrix2x2(double const sAs, complex_t const sAp, double const pAp, complex_t & alpha, double & beta) {
double const sAppAs = std::real(conjugate(sAp)*sAp); 
double const gamma = 0.5*(sAs + pAp) - std::sqrt(0.25*pow2(sAs - pAp) + sAppAs);

if (std::abs((sAs - gamma)/gamma) > 2e-16) {
double const tm2 = sAppAs + pow2(sAs - gamma);
double const tmp = 1.0/std::sqrt(tm2);
alpha = sAp*complex_t(tmp);
beta  = (gamma - sAs)*tmp;
} else {
alpha = 1;
beta  = 0;
}
return gamma;
} 

template <class operator_t>
status_t eigensolve(
typename operator_t::complex_t eigenstates[] 
, double eigenvalues[] 
, int const nbands 
, operator_t const & op 
, int const echo=0 
, float const threshold=1e-8f 
) {
using complex_t = typename operator_t::complex_t; 
using doublecomplex_t = decltype(to_double_complex_t(complex_t(1)));
using real_t = decltype(std::real(complex_t(1))); 

status_t stat(0);
int const maxiter = control::get("conjugate_gradients.max.iter", 132);
int const restart_every_n_iterations = std::max(1, 4);

if (echo > 0) std::printf("# start CG onto %d bands\n", nbands);

doublecomplex_t const dV = op.get_volume_element();
size_t const nv = op.get_degrees_of_freedom();
bool const use_overlap = op.use_overlap();
bool const use_precond = op.use_precond();

int const cg0sd1 = control::get("conjugate_gradients.steepest.descent", 0.); 
bool const use_cg = (0 == cg0sd1);

view2D<complex_t> s(eigenstates, nv); 

view2D<complex_t> Ss(eigenstates, nv); 
if (use_overlap) {
if (echo > 8) std::printf("# %s allocate %.3f MByte for %d adjoint wave functions\n", 
__func__, nbands*nv*1e-6*sizeof(complex_t), nbands);
Ss = view2D<complex_t>(nbands, nv, 0.0); 
} 

int const northo[] = {2, cg0sd1, 1};

int const nsinglevectors = 4 + (use_precond?1:0) + (use_cg?1:0) + (use_overlap?3:0);
if (echo > 8) std::printf("# %s allocate %.3f MByte for %d single vectors\n", 
__func__, nsinglevectors*nv*1e-6*sizeof(complex_t), nsinglevectors);
view2D<complex_t> mem(nsinglevectors, nv, 0.0);
int imem{0};
auto const Hs=mem[imem++], Hcon=mem[imem++], grd=mem[imem++], dir=mem[imem++], 
Pgrd=use_precond  ? mem[imem++] : grd,
SPgrd=use_overlap ? mem[imem++] : Pgrd,
con=use_cg        ? mem[imem++] : Pgrd,
Scon=use_overlap  ? mem[imem++] : con,
Sdir=use_overlap  ? mem[imem++] : dir;
if (echo > 9) std::printf("# CG vectors: Hs=%p, Hcon=%p, grd=%p, dir=%p, \n#   Pgrd=%p, SPgrd=%p, con=%p, Scon=%p, Sdir=%p\n",
(void*)Hs, (void*)Hcon, (void*)grd, (void*)dir, (void*)Pgrd, (void*)SPgrd, (void*)con, (void*)Scon, (void*)Sdir);
if (echo > 8 && nsinglevectors > imem) std::printf("# %s only needed %d of %d single vectors\n", __func__, imem, nsinglevectors);
assert(nsinglevectors >= imem);

std::vector<double> energy(nbands, 0.0), residual(nbands, 9.9e99);

int const num_outer_iterations = control::get("conjugate_gradients.num.outer.iterations", 1.);
for (int outer_iteration = 0; outer_iteration < num_outer_iterations; ++outer_iteration) {
if (echo > 4) std::printf("# CG start outer iteration #%i\n", outer_iteration);

for (int ib_= 0; ib_< nbands; ++ib_) {  int const ib = ib_; 
if (echo > 8) std::printf("\n# start CG for band #%i\n", ib);

if (use_overlap) {
stat += op.Overlapping(Ss[ib], s[ib], echo);
} else assert(Ss[ib] == s[ib]);

for (int iortho = 0; iortho < northo[0]; ++iortho) {
if (echo > 7 && ib > 0) std::printf("# orthogonalize band #%i against %d lower bands\n", ib, ib);
for (int jb = 0; jb < ib; ++jb) {
auto const a = inner_product(nv, s[ib], Ss[jb], dV);
if (echo > 8) std::printf("# orthogonalize band #%i against band #%i with %g %g\n", ib, jb, std::real(a), std::imag(a));
add_product( s[ib], nv,  s[jb], complex_t(-a));
if (use_overlap) {
add_product(Ss[ib], nv, Ss[jb], complex_t(-a));
} else assert(Ss[ib] == s[ib]);
} 

auto const snorm = std::real(inner_product(nv, s[ib], Ss[ib], dV));
if (echo > 7) std::printf("# norm^2 of band #%i is %g\n", ib, snorm);
if (snorm < 1e-12) {
warn("CG failed for band #%i", ib);
return 1;
} 
complex_t const snrm = 1./std::sqrt(snorm);
scale(s[ib], nv, snrm);
if (use_overlap) {
scale(Ss[ib], nv, snrm);
} else assert(Ss[ib] == s[ib]);
} 

double res_old{1};
double prev_energy{0};
bool restart{true};
int last_printf_line{-1};

int it_converged = 0;
for (int iiter = 1; (iiter <= maxiter) && (0 == it_converged); ++iiter) {

if (0 == (iiter - 1) % restart_every_n_iterations) restart = true;
if (restart) {
stat += op.Hamiltonian(Hs, s[ib], echo);
if (use_overlap) {
stat += op.Overlapping(Ss[ib], s[ib], echo);
} else assert(Ss[ib] == s[ib]);

set( dir, nv, complex_t(0)); 
if (use_overlap) {
set(Sdir, nv, complex_t(0)); 
} else assert(Sdir == dir);

restart = false;
} 

energy[ib] = std::real(inner_product(nv, s[ib], Hs, dV));
if (echo > 7) std::printf("# CG energy of band #%i is %g %s in iteration #%i\n", 
ib, energy[ib]*eV, _eV, iiter);

set(grd, nv, Hs, complex_t(-1)); add_product(grd, nv, Ss[ib], complex_t(energy[ib]));
if (echo > 7) std::printf("# norm^2 (no S) of gradient %.e\n", std::real(inner_product(nv, grd, grd, dV)));

if (use_precond) {
op.Conditioner(Pgrd, grd, echo);
} else assert(Pgrd == grd);

for (int iortho = 0; iortho < northo[1]; ++iortho) {
if (echo > 7) std::printf("# orthogonalize gradient against %d bands\n", ib + 1);
for (int jb = 0; jb <= ib; ++jb) {
auto const a = inner_product(nv, Pgrd, Ss[jb], dV);
if (echo > 8) std::printf("# orthogonalize gradient against band #%i with %g %g\n", jb, std::real(a), std::imag(a));
add_product(Pgrd, nv, s[jb], complex_t(-a));
} 
} 

if (use_overlap) {
stat += op.Overlapping(SPgrd, Pgrd, echo);
} else assert(SPgrd == Pgrd);

double const res_new = std::real(inner_product(nv, Pgrd, SPgrd, dV));
if (echo > 7) std::printf("# norm^2 of%s gradient %.e\n", use_precond?" preconditioned":"", res_new);

residual[ib] = std::abs(res_new); 
if (echo > 6) {
if (last_printf_line == __LINE__) std::printf("\r"); else last_printf_line = __LINE__; 
std::printf("# CG band #%i energy %g %s changed %g residual %.2e in iteration #%i", 
ib, energy[ib]*eV, _eV, (energy[ib] - prev_energy)*eV, residual[ib], iiter);
std::fflush(stdout);
} 
prev_energy = energy[ib];

if (use_cg) { 

complex_t const f = ((std::abs(res_new) > tiny<real_t>()) ? res_old/res_new : 0);
if (echo > 7) std::printf("# CG step with old/new = %g/%g = %g\n", res_old, res_new, std::real(f));

scale(dir, nv, f);
add_product(dir, nv, Pgrd, complex_t(1));
set(con, nv, dir); 
if (use_overlap) {
scale(Sdir, nv, f);
add_product(Sdir, nv, SPgrd, complex_t(1));
set(Scon, nv, Sdir); 
} else { assert(Sdir == dir); assert(Scon == con); };

if (echo > 7) std::printf("# norm^2 of con %.e\n", std::real(inner_product(nv, con, Scon, dV)));

for (int iortho = 0; iortho < northo[2]; ++iortho) {
if (echo > 7) std::printf("# orthogonalize conjugate direction against %d bands\n", ib + 1);
for (int jb = 0; jb <= ib; ++jb) {
auto const a = inner_product(nv, con, Ss[jb], dV);
if (echo > 8) std::printf("# orthogonalize conjugate direction against band #%i with %g %g\n", jb, std::real(a), std::imag(a));
add_product(con,  nv,  s[jb], complex_t(-a));
if (use_overlap) {
add_product(Scon, nv, Ss[jb], complex_t(-a));
} else assert(Scon == con);
} 
} 

} else { assert(Scon == SPgrd); assert(con == Pgrd); } 

double const cnorm = std::real(inner_product(nv, con, Scon, dV));
if (echo > 7) std::printf("# norm^2 of conjugate direction is %.e\n", cnorm);
if (cnorm < 1e-12) {
it_converged = iiter; 
} else {
complex_t const cf = 1./std::sqrt(cnorm);

scale(con, nv, cf);
if (use_overlap) {
if (1) {
scale(Scon, nv, cf); 
} else { 
stat += op.Overlapping(Scon, con, echo);
}
} else assert(Scon == con);

stat += op.Hamiltonian(Hcon, con, echo);

double const sHs = energy[ib];
auto   const sHc = inner_product(nv, s[ib], Hcon, dV); 
double const cHc = std::real(inner_product(nv,   con, Hcon, dV));
doublecomplex_t alpha{1};
double          beta{0};
submatrix2x2(sHs, sHc, cHc, alpha, beta);

scale(s[ib], nv, complex_t(alpha)); add_product(s[ib], nv,  con, complex_t(beta));
scale(Hs   , nv, complex_t(alpha)); add_product(Hs   , nv, Hcon, complex_t(beta));
if (use_overlap) {
scale(Ss[ib], nv, complex_t(alpha)); add_product(Ss[ib], nv, Scon, complex_t(beta));
} else assert(Ss[ib] == s[ib]);

if (res_new < threshold) it_converged = -iiter;
} 

res_old = res_new; 
} 

if (maxiter < 1) {
energy[ib] = std::real(inner_product(nv, s[ib], Hs, dV));
} 

if (echo > 4) {
if (last_printf_line > 0) std::printf("\r"); 
std::printf("# CG energy of band #%i is %.9g %s, residual %.1e %s in %d iterations                \n", 
ib, energy[ib]*eV, _eV, residual[ib],
it_converged ? "converged" : "reached", 
it_converged ? std::abs(it_converged) : maxiter);
} 
} 

if (echo > 4) {
char label[64]; std::snprintf(label, 63, "Eigenvalues of outer iteration #%i", outer_iteration);
show_matrix(energy.data(), nbands, 1, nbands, label, eV, '#', 0);
} 

} 

if (eigenvalues) set(eigenvalues, nbands, energy.data()); 

return stat;
} 

#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

#include "particle_in_box.hxx" 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_eigensolve<std::complex<double>>(echo);
stat += test_eigensolve<std::complex<float>> (echo);
stat += test_eigensolve<double>(echo);
stat += test_eigensolve<float> (echo); 
return stat;
} 

#endif 

} 
