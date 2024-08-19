#pragma once

#include "real_space.hxx" 

#include "status.hxx" 

namespace iterative_poisson {

template <typename real_t>
status_t solve(
real_t x[] 
, real_t const b[] 
, real_space::grid_t const & g 
, char const method='M' 
, int const echo=0 
, float const threshold=3e-8 
, float *residual=nullptr 
, int const maxiter=199 
, int const miniter=3  
, int restart=4096 
); 

status_t all_tests(int const echo=0); 

} 
