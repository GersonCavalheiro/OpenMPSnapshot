#pragma once

#include "radial_grid.h" 
#include "quantum_numbers.h" 

#include "status.hxx" 

namespace radial_integrator {

double shoot( 
int const sra 
, radial_grid_t const & g 
, double const rV[] 
, ell_QN_t const ell 
, double const E 
, int & nnodes 
, double* rf=nullptr 
, double* r2rho=nullptr 
); 

template <int SRA> 
int integrate_inwards( 
double gg[] 
, double ff[] 
, radial_grid_t const & g 
, double const rV[] 
, ell_QN_t const ell 
, double const E 
, double const valder[2]=nullptr 
, double *dg=nullptr 
, int *ir_stopped=nullptr 
, int const ir_start=-1 
, int const ir_stop=4 
); 

template <int SRA> 
int integrate_outwards( 
double gg[] 
, double ff[] 
, radial_grid_t const & g 
, double const rV[] 
, ell_QN_t const ell 
, double const E 
, int const ir_stop=-1 
, double *dg=nullptr 
, double const *rp=nullptr 
); 

status_t all_tests(int const echo=0); 

} 
