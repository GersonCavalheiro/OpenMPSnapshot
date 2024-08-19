#pragma once

#include "radial_grid.h" 
#include "quantum_numbers.h" 

#include "status.hxx" 

namespace radial_potential {

double Hartree_potential( 
double rV[] 
, radial_grid_t const & g 
, double const rho4pi[] 
); 

void Hartree_potential(
double vHt[] 
, radial_grid_t const & g 
, double const rho[] 
, int const stride 
, ell_QN_t const ellmax 
, double const q0=0 
); 

status_t all_tests(int const echo=0); 

} 
