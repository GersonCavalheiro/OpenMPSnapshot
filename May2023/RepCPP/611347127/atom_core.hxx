#pragma once

#include <cstdio> 
#include <cmath> 
#include <cassert> 
#include <algorithm> 

#include "radial_grid.h" 
#include "quantum_numbers.h" 

#include "status.hxx" 

namespace atom_core {

status_t solve(
double const Z 
, int const echo=0 
, char const config='a' 
, radial_grid_t const *rg=nullptr 
, double *export_Zeff=nullptr 
); 

status_t scf_atom(
radial_grid_t const & g 
, double const Z 
, int const echo=0 
, double const occupations[][2]=nullptr 
, double *export_Zeff=nullptr
); 

status_t read_Zeff_from_file(
double Zeff[] 
, radial_grid_t const & g 
, double const Z 
, char const *basename="pot/Zeff" 
, double const factor=1 
, int const echo=0 
, char const *prefix="" 
); 

status_t store_Zeff_to_file(
double const Zeff[] 
, double const r[] 
, int const nr 
, double const Z 
, char const *basename="pot/Zeff" 
, double const factor=1 
, int const echo=0 
, char const *prefix="" 
); 

inline void get_Zeff_file_name(
char *filename 
, char const *basename 
, float const Z 
, size_t const nchars=128
) {
std::snprintf(filename, nchars, "%s.%03g", basename, Z);
} 

void rad_pot(
double rV[] 
, radial_grid_t const & g 
, double const rho4pi[] 
, double const Z=0 
, double *energies=nullptr 
); 

inline double guess_energy(double const Z, int const enn) {
auto const Zn2 = (Z*Z)/double(enn*enn);
return -.5*Zn2 *  
(.783517 + 2.5791E-5*Zn2) * 
std::exp(-.01*(enn - 1)*Z);
} 

inline int nl_index(int const enn, int const ell) {
assert(ell >= 0 && "angular momentum quantum number");
assert(enn > ell && "atomic quantum numbers");
return (enn*(enn - 1))/2 + ell;
} 

double neutral_atom_total_energy(double const Z); 

status_t all_tests(int const echo=0); 

} 
