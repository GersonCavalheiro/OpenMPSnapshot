#pragma once

#include <cstdio> 
#include <complex> 
#include <vector> 
#include <algorithm> 
#include <cmath> 

#include "status.hxx" 
#include "data_view.hxx" 
#include "linear_algebra.hxx" 
#include "inline_math.hxx" 
#include "complex_tools.hxx" 
#include "display_units.h" 
#include "print_tools.hxx" 
#include "recorded_warnings.hxx" 

#ifndef NO_UNIT_TESTS
#include "simple_math.hxx" 
#include "grid_operators.hxx" 
#endif

namespace davidson_solver {

template <typename doublecomplex_t, typename complex_t>
void inner_products(
doublecomplex_t s[] 
, int const stride 
, size_t const ndof 
, complex_t const bra[] 
, int const nstates  
, complex_t const ket[] 
, int const mstates  
, double const factor=1
) {
assert(stride >= mstates);
for (int ibra = 0; ibra < nstates; ++ibra) {
auto const bra_ptr = &bra[ibra*ndof];
for (int jket = 0; jket < mstates; ++jket) {
auto const ket_ptr = &ket[jket*ndof];
doublecomplex_t tmp(0);
for (size_t dof = 0; dof < ndof; ++dof) {
tmp += doublecomplex_t(conjugate(bra_ptr[dof])) * doublecomplex_t(ket_ptr[dof]);
} 
s[ibra*stride + jket] = tmp*factor; 
} 
#ifdef NEVER
std::printf("\n# davidson_solver: inner_products: coeffs (%i,:) ", ibra);
printf_vector("%g ", &s[ibra*stride], mstates);
#endif 
} 
} 


template <typename complex_t>
void vector_norm2s(
double s[] 
, size_t const ndof
, complex_t const ket[] 
, int const mstates  
, complex_t const *bra=nullptr
, double const factor=1
) {
for (int jket = 0; jket < mstates; ++jket) {
auto const ket_ptr = &ket[jket*ndof];
auto const bra_ptr = bra ? &bra[jket*ndof] : ket_ptr;
double tmp{0};
for (size_t dof = 0; dof < ndof; ++dof) {
tmp += std::real(conjugate(bra_ptr[dof]) * ket_ptr[dof]);
} 
s[jket] = tmp*factor; 
} 
} 


template <typename real_t>
void show_matrix(
real_t const mat[]
, int const stride
, int const n
, int const m
, char const *name=nullptr
, double const unit=1
, char const *_unit="1"
) {
if (n < 1) return;
if (is_complex<real_t>()) return;
if (1 == n) {
std::printf("# Vector=%s (%s)", name, _unit);
} else {
std::printf("\n# %dx%d Matrix=%s (%s)\n", n, m, name, _unit);
} 
for (int i = 0; i < n; ++i) {
if (n > 1) std::printf("#%4i ", i);
for (int j = 0; j < m; ++j) {
std::printf((1 == n)?" %.3f":" %7.3f", std::real(mat[i*stride + j])*unit);
} 
std::printf("\n");
} 
std::printf("\n");
} 

template <class operator_t>
status_t eigensolve(
typename operator_t::complex_t waves[] 
, double energies[] 
, int const nbands 
, operator_t const & op
, int const echo=0 
, float const mbasis=2
, int const niterations=2
, float const threshold=1e-4
, char const * func="Davidson"
) {
using complex_t = typename operator_t::complex_t; 
using doublecomplex_t = decltype(to_double_complex_t(complex_t(1))); 
using real_t = decltype(std::real(complex_t(1)));
status_t stat(0);
if (nbands < 1) return stat;

double const dV = op.get_volume_element();
size_t const ndof = op.get_degrees_of_freedom(); 

int const max_space = std::ceil(mbasis*nbands);
int sub_space{nbands}; 
if (echo > 7) std::printf("# start %s with %d bands, subspace size up to %d bands\n", func, sub_space, max_space);

double const threshold2 = pow2(threshold); 

auto const op_echo = echo - 16; 

view3D<doublecomplex_t> matrices(2, max_space, max_space);
auto Hmt = matrices[0], Ovl = matrices[1]; 
std::vector<double> eigval(max_space);
std::vector<double> residual_norm2s(max_space);

complex_t const zero(0);
view2D<complex_t>  psi(max_space, ndof, zero); 
view2D<complex_t> hpsi(max_space, ndof, zero); 
view2D<complex_t> spsi(max_space, ndof, zero); 
view2D<complex_t> epsi(max_space, ndof, zero); 

set(psi.data(), nbands*ndof, waves); 

int niter{niterations};
for (int iteration = 0; iteration < niter; ++iteration) {
if (echo > 9) std::printf("# %s iteration %i\n", func, iteration);

int n_drop{0};
do {
for (int istate = 0; istate < sub_space; ++istate) {
stat += op.Hamiltonian(hpsi[istate], psi[istate], op_echo);
stat += op.Overlapping(spsi[istate], psi[istate], op_echo);
} 

inner_products(Ovl.data(), Ovl.stride(), ndof, psi.data(), sub_space, spsi.data(), sub_space, dV);
inner_products(Hmt.data(), Hmt.stride(), ndof, psi.data(), sub_space, hpsi.data(), sub_space, dV);

if (echo > 9) show_matrix(Ovl.data(), Ovl.stride(), sub_space, sub_space, "Overlap");
if (echo > 8) show_matrix(Hmt.data(), Hmt.stride(), sub_space, sub_space, "Hamiltonian", eV, _eV);

if (0) { 
view2D<doublecomplex_t> Ovl_copy(sub_space, sub_space, doublecomplex_t(0));
for (int i = 0; i < sub_space; ++i) {
set(Ovl_copy[i], sub_space, Ovl[i]); 
} 

if (1) { 
double dev2{0};
for (int i = 0; i < sub_space; ++i) {
for (int j = 0; j < i; ++j) {
dev2 = std::max(dev2, std::norm(Ovl_copy(i,j) - conjugate(Ovl_copy(j,i))));
} 
} 
auto const dev = std::sqrt(std::max(0.0, dev2)); 
if (echo > 9 && dev > 1e-14) std::printf("# %s: the %d x %d overlap matrix deviates from %s by %.1e\n", func,
sub_space, sub_space, is_complex<doublecomplex_t>() ? "Hermitian" : "symmetric", dev);
if (dev > 1e-12) warn("the overlap matrix deviates by %.1e from symmetric/Hermitian", dev);
} 

auto const info = linear_algebra::eigenvalues(eigval.data(), sub_space, Ovl_copy.data(), Ovl_copy.stride());
if (1) {
std::printf("# %s: lowest eigenvalues of the %d x %d overlap matrix ", func, sub_space, sub_space);
for (int i = 0; i < std::min(9, sub_space) - 1; ++i) {
std::printf(" %.3g", eigval[i]);
} 
if (sub_space > 9) std::printf(" ...");
std::printf(" %g", eigval[sub_space - 1]);
if (0 != info) std::printf(", info= %i\n", int(info));
std::printf("\n");
} 
if (eigval[0] <= 0.0) {
warn("overlap matrix is not positive definite, lowest eigenvalue is %g", eigval[0]);
} 

int drop_bands{0}; while (eigval[drop_bands] < 1e-4) ++drop_bands;
n_drop = drop_bands;
if (n_drop > 0) {
if (echo > 0) std::printf("# %s: drop %d bands to stabilize the overlap\n", func, n_drop);
for (int i = 0; i < sub_space - n_drop; ++i) {
int const ii = i + n_drop;
set(epsi[i], ndof, zero);
for (int j = 0; j < sub_space; ++j) {
add_product(epsi[i], ndof, psi[j], complex_t(Ovl_copy(ii,j)));
} 
} 
std::swap(psi, epsi); 
sub_space -= n_drop;
} 

} 
} while(n_drop > 0); 

auto const info = linear_algebra::eigenvalues(eigval.data(), sub_space, Hmt.data(), Hmt.stride(), Ovl.data(), Ovl.stride());
if (info) {
warn("generalized eigenvalue problem returned INFO=%i", info);
stat += info;
} else {
auto const & eigvec = Hmt; 
if (echo > 8) show_matrix(eigval.data(), 0, 1, sub_space, "Eigenvalues", eV, _eV);

for (int i = 0; i < sub_space; ++i) {
set(epsi[i], ndof, zero);
for (int j = 0; j < sub_space; ++j) {
add_product(epsi[i], ndof, psi[j], complex_t(eigvec(i,j)));
} 
} 
std::swap(psi, epsi); 

if (sub_space < max_space) {

bool const with_overlap = true;
for (int i = 0; i < sub_space; ++i) {
stat += op.Hamiltonian(hpsi[i], psi[i], op_echo);
set(epsi[i], ndof, hpsi[i]);
stat += op.Overlapping(spsi[i], psi[i], op_echo);
add_product(epsi[i], ndof, spsi[i], complex_t(-eigval[i]));

if (with_overlap) stat += op.Overlapping(spsi[i], epsi[i], op_echo);
} 
vector_norm2s(residual_norm2s.data(), ndof, epsi.data(), sub_space, 
with_overlap ? spsi.data() : nullptr, dV);
#ifdef DEBUG
std::printf("# Davidson: unsorted residual norms^2 ");
printf_vector(" %.1e", residual_norm2s.data(), sub_space);
#endif 

std::vector<double> res_norm2_sort(sub_space);              
set(res_norm2_sort.data(), sub_space, residual_norm2s.data()); 
std::sort(res_norm2_sort.rbegin(), res_norm2_sort.rend()); 
#ifdef DEBUG
std::printf("# Davidson: largest residual norms^2 ");
printf_vector(" %.1e", res_norm2_sort.data(), sub_space);
#endif 

int const max_bands = max_space - sub_space; 
int new_bands{0};
double thres2{9.999e99};
for (int i = 0; i < sub_space; ++i) {
auto const rn2 = res_norm2_sort[i];
if (rn2 > threshold2) {
if (new_bands < max_bands) {
++new_bands;
thres2 = std::min(thres2, rn2);
} 
} 
} 
int const add_bands = new_bands;
if (echo > 0) {
std::printf("# Davidson: in iteration #%i add %d residual vectors", iteration, add_bands);
if (add_bands > 0) std::printf(" with norm2 above %.3e", thres2);
std::printf("\n");
} 

std::vector<short> indices(add_bands);
int new_band{0};
for (int i = 0; i < sub_space; ++i) {
auto const rn2 = residual_norm2s[i];
if (rn2 >= thres2) {
if (new_band < max_bands) {
indices[new_band] = i;
++new_band;
} 
} 
} 

if (echo > 9 && add_bands > 0) {
std::printf("# Davidson: add %d residual vectors: ", add_bands);
printf_vector(" %i", indices.data(), new_band);
} 
if (new_band != add_bands) error("new_bands=%d != %d=add_bands", new_band, add_bands);

for (int i = 0; i < add_bands; ++i) {
int const j = indices[i];
int const ii = sub_space + i;
real_t const f = 1./std::sqrt(residual_norm2s[j]);
set(psi[ii], ndof, epsi[j], f); 
} 

sub_space += add_bands; 

niter = iteration + 2 * (add_bands > 0);
} else { 
niter = iteration; 
} 

} 

} 

set(waves, nbands*ndof, psi.data());  
set(energies, nbands, eigval.data()); 

if (echo > 4) show_matrix(eigval.data(), 0, 1, sub_space, "Eigenvalues", eV, _eV);

return stat;
} 

template <class operator_t>
status_t rotate(
typename operator_t::complex_t waves[] 
, double energies[] 
, int const nbands 
, operator_t const & op 
, int const echo=0 
) {
return eigensolve(waves, energies, nbands, op, echo, 1, 1, 0.f, __func__);
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
