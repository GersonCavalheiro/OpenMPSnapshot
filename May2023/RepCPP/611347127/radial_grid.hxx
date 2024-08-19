#pragma once

#include <cmath> 

#include "radial_grid.h" 
#include "status.hxx" 

namespace radial_grid {

double constexpr default_anisotropy = 0.01;
float  constexpr default_Rmax = 9.45;

char constexpr equation_exponential = 'e';
char constexpr equation_equidistant = '=';
char constexpr equation_reciprocal  = '/';

radial_grid_t* create_radial_grid( 
int const npoints 
, float const rmax=default_Rmax 
, char equation='\0' 
, double const anisotropy=default_anisotropy 
); 

radial_grid_t* create_pseudo_radial_grid(
radial_grid_t const & tru 
, double const r_min=1e-3 
, int const echo=0 
); 

inline radial_grid_t* create_default_radial_grid(float const Z_protons=0
, float const rmax=default_Rmax) {
return create_radial_grid(250*std::sqrt(std::abs(Z_protons) + 9), rmax);
} 

void destroy_radial_grid(radial_grid_t* g, char const *name=""); 

int find_grid_index(radial_grid_t const & g, double const radius); 

double get_prefactor(radial_grid_t const & g); 

char const* get_formula(char const equation='\0'); 

status_t all_tests(int const echo=0); 

} 
