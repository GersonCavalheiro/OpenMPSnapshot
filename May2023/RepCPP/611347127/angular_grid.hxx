#pragma once

#include <vector> 

#include "status.hxx" 
#include "gaunt_entry.h" 

namespace angular_grid {

int get_grid_size(int const ellmax, int const echo=0); 

status_t transform(
double out[] 
, double const in[] 
, int const stride 
, int const ellmax 
, bool const back=false 
, int const echo=0 
); 

std::vector<gaunt_entry_t> create_numerical_Gaunt(int const ellmax, int const echo=0); 

void cleanup(int const echo=0); 

status_t all_tests(int const echo=0); 

} 
