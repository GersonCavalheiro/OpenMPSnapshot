#pragma once

#include <cstdio> 
#include <cassert> 
#include <vector> 
#include <algorithm> 

#include "status.hxx" 
#include "control.hxx" 
#include "atom_core.hxx" 
#include "quantum_numbers.h" 
#include "radial_grid.hxx" 
#include "sigma_config.hxx" 
#include "spherical_state.hxx" 
#include "chemical_symbol.h" 
#include "recorded_warnings.hxx" 
#include "radial_eigensolver.hxx" 
#include "inline_math.hxx" 
#include "sho_tools.hxx" 

namespace element_config {

inline char const* element_symbol(int const Z) {
int const z128 = Z & 127; 
return &(element_symbols[2*z128]); 
} 

char const ellchar[] = "spdfgh+";

inline sigma_config::element_t const & get(
double const Z_core 
, double const ionization
, radial_grid_t const & rg 
, double const rV[]   
, char const *const Sy="X"
, double const core_state_localization=-1
, int const echo=3 
, int const SRA=1 
) {
if (echo > 0) std::printf("# %s element_config for Z= %g\n", Sy, Z_core);

auto & e = *(new sigma_config::element_t); 

e.Z = Z_core;

char Sy_config[32];
std::snprintf(Sy_config, 31, "element_%s.rcut", Sy);
auto const rcut_default = control::get("element_config.rcut", 2.0);
e.rcut = control::get(Sy_config, rcut_default); 

std::snprintf(Sy_config, 31, "element_%s.sigma", Sy);
auto const sigma_default = control::get("element_config.sigma", 0.5);
e.sigma = control::get(Sy_config, sigma_default); 

std::snprintf(Sy_config, 31, "element_%s.numax", Sy);
auto const numax_default = control::get("element_config.numax", 3.);
e.numax = int(control::get(Sy_config, numax_default));

for (int ell = 0; ell < 8; ++ell) {
e.nn[ell] = std::max(0, sho_tools::nn_max(e.numax, ell));
} 

double hole_charge{0}, hole_charge_used{0};
int inl_hole{-1}, ell_hole{-1}; 
std::snprintf(Sy_config, 31, "element_%s.hole.enn", Sy);
int const enn_hole = control::get(Sy_config, 0.);
if (enn_hole > 0) {
if (enn_hole < 9) {
std::snprintf(Sy_config, 31, "element_%s.hole.ell", Sy);
ell_hole = control::get(Sy_config, -1.);
if (ell_hole >= 0 && ell_hole < enn_hole) {
inl_hole = atom_core::nl_index(enn_hole, ell_hole);
std::snprintf(Sy_config, 31, "element_%s.hole.charge", Sy);
hole_charge = std::min(std::max(0.0, control::get(Sy_config, 1.)), 2.*(2*ell_hole + 1));
} else warn("%s=%d is out of range [0, %d]", Sy_config, ell_hole, enn_hole - 1);
} else warn("%s=%d is too large", Sy_config, enn_hole);
}

set(e.method, 16, '\0'); 
auto const method_default = control::get("element_config.method", "sinc");
std::snprintf(Sy_config, 31, "element_%s.method", Sy);
std::snprintf(e.method, 15, "%s", control::get(Sy_config, method_default));

set(e.occ[0], 32*2, 0.0); 

double core_valence_separation{0}, core_semicore_separation{0}, semi_valence_separation{0};
if (core_state_localization > 0) {
if (echo > 0) std::printf("# %s use core state localization criterion with %g %%\n", Sy, core_state_localization*100);
} else {
core_valence_separation  = control::get("element_config.core.valence", -2.0); 
core_semicore_separation = control::get("element_config.core.semicore",    core_valence_separation);
semi_valence_separation  = control::get("element_config.semicore.valence", core_valence_separation);
if (core_semicore_separation > semi_valence_separation) {
warn("%s element_config.core.semicore=%g %s may not be higher than ~.semicore.valence=%g %s, correct for it",
Sy, core_semicore_separation*eV, _eV, semi_valence_separation*eV, _eV);
core_semicore_separation = semi_valence_separation; 
} 
} 

int const ir_cut = radial_grid::find_grid_index(rg, e.rcut);
if (echo > 6) std::printf("# %s cutoff radius %g %s at grid point %d of max. %d\n",
Sy, e.rcut*Ang, _Ang, ir_cut, rg.n);

std::vector<int8_t> as_valence(40, -1);
std::vector<enn_QN_t> enn_core_ell(8, 0); 
std::vector<double> wave(rg.n, 0.0); 
std::vector<double> r2rho(rg.n, 0.0);
double csv_charge[3] = {0, 0, 0};

{ 
int highest_occupied_state_index{-1};
int ics{0}; 

double n_electrons{Z_core + ionization}; 
for (int nq_aux = 0; nq_aux < 8; ++nq_aux) {    
enn_QN_t enn = (nq_aux + 1)/2;              
for (int ell = nq_aux/2; ell >= 0; --ell) { 
++enn; 
{   int const jj = 2*ell;
double const max_occ = 2*(jj + 1); 

char tag[4]; std::snprintf(tag, 3, "%d%c", enn, ellchar[ell]);
set(r2rho.data(), rg.n, 0.0); 

double E{atom_core::guess_energy(Z_core, enn)}; 
radial_eigensolver::shooting_method(SRA, rg, rV, enn, ell, E, wave.data(), r2rho.data());

int const inl = atom_core::nl_index(enn, ell);

double const hole = hole_charge*(inl_hole == inl);
double const occ_no_hole = std::min(std::max(0., n_electrons),        max_occ);
double const occ         = std::min(std::max(0., occ_no_hole - hole), max_occ);
double const real_hole_charge = occ_no_hole - occ;
if (real_hole_charge > 0) {
hole_charge_used = real_hole_charge;
if (echo > 1) std::printf("# %s %s has an occupation hole of element_%s.hole.charge=%g electrons\n", 
Sy, tag,                                Sy, real_hole_charge);
assert(enn_hole == enn && ell_hole == ell);
} 

int csv{csv_undefined};
auto const charge_outside = show_state_analysis(echo, Sy, rg, wave.data(), tag, occ, E, "?", ir_cut);
if (core_state_localization > 0) {
if (charge_outside > core_state_localization) {
csv = valence; 
} else {
csv = core; 
} 
} else { 
if (E > semi_valence_separation) {
csv = valence; 
} else if (E > core_semicore_separation) {
csv = semicore; 
} else { 
csv = core; 
} 
} 
assert(csv_undefined != csv);
if (echo > 15) std::printf("# as_%s[nl_index(enn=%d, ell=%d) = %d] = %d\n", csv_name(csv), enn, ell, inl, ics);

if (valence == csv) as_valence[inl] = ics; 

if (occ > 0) {
highest_occupied_state_index = ics; 
if (echo > 5) show_state(Sy, csv_name(csv), tag, occ, E);
if (as_valence[inl] < 0) {
enn_core_ell[ell] = std::max(enn, enn_core_ell[ell]); 
} 
csv_charge[csv] += occ;

double const has_norm = dot_product(rg.n, r2rho.data(), rg.dr);
if (has_norm <= 0) {
warn("%s %i%c-state cannot be normalized! integral= %g electrons", Sy, enn, ellchar[ell], has_norm);
} 

e.occ[inl][0] = e.occ[inl][1] = 0.5*occ; 
e.csv[inl] = csv;
} 

n_electrons -= occ; 
++ics;
} 
} 
} 
int const nstates = highest_occupied_state_index + 1; 
if (echo > 0) std::printf("# %s found %d spherical states\n", Sy, nstates);

if (n_electrons > 0) warn("%s after distributing %g, %g electrons remain",
Sy, Z_core - ionization, n_electrons);

auto const total_n_electrons = csv_charge[core] + csv_charge[semicore] + csv_charge[valence];
if (echo > 2) std::printf("# %s initial occupation with %g electrons: %g core, %g semicore and %g valence electrons\n", 
Sy, total_n_electrons, csv_charge[core], csv_charge[semicore], csv_charge[valence]);

if (inl_hole >= 0) {
auto const diff = hole_charge_used - hole_charge;
if (std::abs(diff) > 5e-16) {
warn("hole.charge=%g requested in %s-%d%c state but used %g electrons (difference %.1e)",
hole_charge, Sy, enn_hole,ellchar[ell_hole], hole_charge_used, diff);
} 
if (echo > 0) std::printf("# %s occupation hole of %g electrons in the %d%c state\n",
Sy, hole_charge_used, enn_hole,ellchar[ell_hole]);
} 

} 

return e;
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_show_all_elements(int const echo=2) {
if (echo > 0) std::printf("# %s:  %s\n", __FILE__, __func__);
float iocc[32];

for (int spin = 1; spin >= 0; --spin) { 
auto const echo_occ = 9; 
if (echo > 2 + 3*spin) { 
if (spin > 0) std::printf("# including spin-orbit\n");
std::printf("\n#  nl    j   occ    index\n");
for (int inl = 0; inl < 32; ++inl) iocc[inl] = 0; 
int iZ{0}; 
int ishell{0}; 
for (int nq_aux = 0; nq_aux < 8; ++nq_aux) {    
int enn{(nq_aux + 1)/2};                    
for (int ell = nq_aux/2; ell >= 0; --ell) { 
++enn;
for (int jj = 2*ell + spin; jj >= std::max(0, 2*ell - spin); jj -= 2) { 
std::printf("%4d%c%6.1f%4d%9d    %c", enn, ellchar[ell], jj*.5f,
(2 - spin)*(jj + 1), ishell, (echo > echo_occ)?'\n':' ');
for (int mm = -jj; mm <= jj; mm += 2) { 
for (int s = 0; s <= 1 - spin; ++s) { 
iocc[ishell] += 1;
++iZ; 
auto const El = element_symbol(iZ);
if (echo > echo_occ) {
std::printf("# %c%c occupation", El[0], El[1]);
for (int inl = 0; inl <= ishell; ++inl) {
std::printf(" %g", iocc[inl]);
} 
std::printf("\n");
} else {
std::printf(" %c%c", El[0], El[1]);
} 
} 
} 
++ishell; 
std::printf("\n");
} 
} 
} 
double sum_occ{0};
for (int inl = 0; inl < ishell; ++inl) {
sum_occ += iocc[inl];
} 
std::printf("# sum(occ) = %.3f\n", sum_occ);
assert(120 == sum_occ); 
assert(20 + 12*spin == ishell); 
} 
} 
if (echo > 0) std::printf("# table of elements according to Aco Z. Muradjan\n\n");
return 0;
} 


inline status_t all_tests(int const echo=0) {
return test_show_all_elements(echo);
} 

#endif 

} 
