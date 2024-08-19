#pragma once

#include <cstdio> 
#include <complex> 

#include "status.hxx" 
#include "grid_operators.hxx" 
#include "complex_tools.hxx" 
#include "sho_tools.hxx" 
#include "sho_projection.hxx" 
#include "data_view.hxx" 
#include "data_list.hxx" 
#include "inline_math.hxx" 
#include "control.hxx" 
#include "display_units.h" 
#include "fermi_distribution.hxx" 
#include "print_tools.hxx" 

namespace density_generator {

template <typename complex_t>
void add_to_density(
double rho[] 
, size_t const nzyx 
, complex_t const wave[] 
, double const weight=1 
, int const echo=0 
, int const iband=-1 
, int const ikpoint=-1 
) {
for (size_t izyx = 0; izyx < nzyx; ++izyx) { 
rho[izyx] += weight * std::norm(wave[izyx]);
} 
} 


template <typename complex_t>
void add_to_density_matrices(
double *const atom_rho[] 
, complex_t const atom_coeff[] 
, uint32_t const coeff_starts[] 
, int const natoms 
, double const weight=1 
, int const echo=0 
, int const iband=-1 
, int const ikpoint=-1 
) {
for (int ia = 0; ia < natoms; ++ia) {
int const offset = coeff_starts[ia];
int const ncoeff = coeff_starts[ia + 1] - offset;
auto const a_rho = atom_rho[ia];
for (int i = 0; i < ncoeff; ++i) {
auto const c_i = conjugate(atom_coeff[offset + i]);
#ifdef DEVEL
if (echo > 16) std::printf("# k-point #%i band #%i atom #%i coeff[%i]= %.6e %g |c|= %g\n",
ikpoint, iband, ia, i, std::real(c_i), std::imag(c_i), std::abs(c_i));
#endif 
for (int j = 0; j < ncoeff; ++j) {
auto const c_j = atom_coeff[offset + j];
a_rho[i*ncoeff + j] += weight * std::real(c_i * c_j);
} 
} 
} 
} 


template <typename uint_t>
std::vector<uint32_t> prefetch_sum(std::vector<uint_t> const & natom_coeff) {
int const na = natom_coeff.size();
std::vector<uint32_t> coeff_starts(na + 1);
coeff_starts[0] = 0u;
for (int ia = 0; ia < na; ++ia) {
coeff_starts[ia + 1] = coeff_starts[ia] + natom_coeff[ia];
} 
return coeff_starts;
} 


template <class grid_operator_t>
view2D<typename grid_operator_t::complex_t> atom_coefficients(
std::vector<uint32_t> & coeff_starts
, typename grid_operator_t::complex_t const eigenfunctions[]
, grid_operator_t const & op
, int const nbands=1
, int const echo=0 
, int const ikpoint=-1 
) {
using complex_t = typename grid_operator_t::complex_t; 
using real_t    = decltype(std::real(complex_t(1)));

status_t stat(0);

assert(eigenfunctions); 

auto const natoms = op.get_natoms();
auto const nzyx   = op.get_degrees_of_freedom();
if (echo > 3) std::printf("# %s assume %d atoms and %d bands with %ld grid elements\n", __func__, natoms, nbands, nzyx);

std::vector<std::vector<real_t>> scale_factors(natoms);
std::vector<uint16_t> natom_coeff(natoms);
for (int ia = 0; ia < natoms; ++ia) {
int const numax = op.get_numax(ia);
natom_coeff[ia] = sho_tools::nSHO(numax);
auto const sigma = op.get_sigma(ia);
if (echo > 6) std::printf("# %s atom #%i has numax=%d and %d coefficients, sigma= %g %s\n",
__func__, ia, numax, natom_coeff[ia], sigma*Ang,_Ang);
assert(sigma > 0);
scale_factors[ia] = sho_projection::get_sho_prefactors<real_t>(numax, sigma);
} 

coeff_starts = prefetch_sum(natom_coeff);
int const n_all_coeff = coeff_starts[natoms];
int const stride = align<0>(n_all_coeff); 
if (echo > 3) std::printf("# %s %d atoms have %d coefficients, stride %d\n", __func__, natoms, n_all_coeff, stride);
if (echo > 3) std::printf("# %s coefficients(%d,%d of %d)\n", __func__, nbands, n_all_coeff, stride);
view2D<complex_t> coeff(nbands, stride, complex_t(0)); 

if (echo > 3) std::printf("# %s assume psi(%d,%ld)\n", __func__, nbands, nzyx);
view2D<complex_t const> const psi_k(eigenfunctions, nzyx); 

data_list<complex_t> atom_coeff(natom_coeff); 

{ 
for (int iband = 0; iband < nbands; ++iband) {

stat += op.get_atom_coeffs(atom_coeff.data(), psi_k[iband], echo/2); 

for (int ia = 0; ia < natoms; ++ia) {
auto const c_ia = &coeff(iband,coeff_starts[ia]);
product(c_ia, natom_coeff[ia], atom_coeff[ia], scale_factors[ia].data());
#ifdef DEVEL
for (int i = 0; i < natom_coeff[ia]; ++i) {
auto const c_i = c_ia[i];
if (echo > 16) std::printf("# k-point #%i band #%i atom #%i coeff[%i]= %.6e %g new\n",
ikpoint, iband, ia, i, std::real(c_i), std::imag(c_i));
} 
#endif 
} 

} 

} 

if (int(stat) && echo > 3) std::printf("# %s get_atom_coeffs produced status sum= %i\n", __func__, int(stat));
return coeff;
} 


template <typename complex_t>
status_t density(
double *const rho              
, double *const *const atom_rho  
, fermi_distribution::FermiLevel_t & Fermi
, double const eigenenergies[]     
, complex_t const eigenfunctions[] 
, complex_t const atom_coeff[]     
, uint32_t const coeff_starts[]    
, int const natoms
, real_space::grid_t const & g
, int const nbands=1
, double const weight_k=1
, int const echo=0 
, int const ikpoint=-1 
, double *const d_rho=nullptr             
, double *const *const d_atom_rho=nullptr 
, double charges[]=nullptr 
) {

status_t stat(0);

if (nullptr == eigenfunctions) { warn("no eigenfunctions received, expect %d", nbands); return -1; }

if (echo > 3) std::printf("# %s assume %d bands on a %d x %d x %d grid\n", __func__, nbands, g('x'), g('y'), g('z'));

auto const n_all_coeff = coeff_starts ? coeff_starts[natoms] : 0;
if (echo > 3) std::printf("# %s assume psi(%d,%ld)\n", __func__, nbands, g.all());
view2D<complex_t const> const psi_k(eigenfunctions, g.all()); 
if (echo > 3) std::printf("# %s assume atom_coeff with stride %d\n", __func__, n_all_coeff);
view2D<complex_t const> const a_coeff(atom_coeff, n_all_coeff); 

double constexpr occ_threshold = 1e-16;
double const kT = Fermi.get_temperature(); 
int const spinfactor = Fermi.get_spinfactor(); 
if (echo > 3) std::printf("# %s spin factor %d and temperature %g %s\n", __func__, spinfactor, kT*Kelvin, _Kelvin);

double charge[3] = {0, 0, 0}; 

{ 
double const weight_sk = spinfactor * weight_k;

std::vector<double> occupation(nbands, 0.0);
std::vector<double> d_occupation(nbands, 0.0); 
Fermi.get_occupations(occupation.data(), eigenenergies, nbands, weight_k, echo, d_occupation.data());

int ilub{nbands}; 
double charge_k[3] = {1, 0, 0}; 
for (int iband = 0; iband < nbands; ++iband) {
auto const psi_nk = psi_k[iband];

if (occupation[iband] >= occ_threshold) {
charge_k[1] += occupation[iband]*spinfactor;
double const weight_nk = occupation[iband] * weight_sk;
if (echo > 6) std::printf("# %s: k-point #%i bands #%i \toccupation= %.6f d_occ= %g E= %g %s\n",
__func__, ikpoint, iband, occupation[iband], d_occupation[iband]*kT, eigenenergies[iband]*eV, _eV);

add_to_density(rho, g.all(), psi_nk, weight_nk, echo, iband, ikpoint);
#if 0 
if (echo > 0) { 
std::printf("# valence density ");
auto const new_charge = print_stats(rho, g.all(), g.dV());
std::printf("# valence density of band #%i added %g electrons\n", iband, new_charge - old_charge);
old_charge = new_charge;
} 
#endif 
add_to_density_matrices(atom_rho, a_coeff[iband],
coeff_starts, natoms, weight_nk, echo, iband, ikpoint);

if (d_occupation[iband] >= occ_threshold) {
charge_k[2] += d_occupation[iband]*spinfactor;
double const d_weight_nk = d_occupation[iband] * weight_sk;

if (d_rho) {
add_to_density(d_rho, g.all(), psi_nk, d_weight_nk, echo, iband, ikpoint);
} 

if (d_atom_rho) {
add_to_density_matrices(d_atom_rho, a_coeff[iband],
coeff_starts, natoms, d_weight_nk, echo, iband, ikpoint);
} 

} 

} else {
ilub = std::min(ilub, iband);
} 

} 
if (ilub < nbands) {
if (echo > 6) std::printf("# %s: k-point #%i band #%i at %g %s and above did not"
" contribute to the density\n", __func__, ikpoint, ilub, eigenenergies[ilub]*eV,_eV);
} 
if (echo > 5) std::printf("# %s: k-point #%i has charge %g electrons and derivative %g\n",
__func__, ikpoint, charge_k[1], charge_k[2]*kT); 
add_product(charge, 3, charge_k, weight_k);
} 
if (charges) add_product(charges, 3, charge, 1.0);
if (echo > 1) { std::printf("\n# Total valence density "); print_stats(rho, g.all(), g.dV()); }
if (d_rho) {
if (echo > 3) { std::printf("# Total response density"); print_stats(d_rho, g.all(), g.dV(), "", kT); }
} 
#if 1
if (echo > 6) {
for (int ia = 0; ia < natoms; ++ia) {
int const ncoeff = coeff_starts[ia + 1] - coeff_starts[ia];
double max_rho{1e-12}; for (int ij = 0; ij < pow2(ncoeff); ++ij) max_rho = std::max(max_rho, atom_rho[ia][ij]);
std::printf("\n# show %d x %d density matrix for atom #%i in %s-order, normalized to maximum %.6e\n", 
ncoeff, ncoeff, ia, sho_tools::SHO_order2string(sho_tools::order_zyx).c_str(), max_rho);
for (int i = 0; i < ncoeff; ++i) {
std::printf("# izyx=%d\t", i);
for (int j = 0; j < ncoeff; ++j) {
std::printf(" %9.6f", atom_rho[ia][i*ncoeff + j]/max_rho);
} 
std::printf("\n");
} 
std::printf("\n");
} 
} 
#endif 

return stat;
} 

#ifdef NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_init(int const echo=3) {
real_space::grid_t const g(4, 5, 6);
grid_operators::grid_operator_t<float,double> const op(g, grid_operators::empty_list_of_atoms());
std::vector<float> wave(g.all());
std::vector<double> rho(g.all());
std::iota(wave.begin(), wave.end(), 0);
fermi_distribution::FermiLevel_t eF(1);
double spectrum[] = {0};
return density(rho.data(), nullptr, eF, spectrum, wave.data(),
wave.data(), 
nullptr, op.get_natoms(), g, 1, 1, echo, -1, nullptr, nullptr);
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_init(echo);
return stat;
} 

#endif 

} 
