#pragma once

#include "quantum_numbers.h" 
#include "radial_grid.h" 

#include "status.hxx" 

namespace radial_eigensolver {

status_t shooting_method(
int const sra 
, radial_grid_t const & g 
, double const rV[] 
, enn_QN_t const enn 
, ell_QN_t const ell 
, double & E 
, double* rf=nullptr 
, double* r2rho=nullptr 
, int const maxiter=999 
, float const threshold=1e-15 
); 

status_t all_tests(int const echo=0); 

} 
