#pragma once

#include <vector> 

#include "status.hxx" 
#include "data_view.hxx" 

namespace sho_basis {

status_t load(
std::vector<view2D<double>> & basis
, std::vector<int> & indirection
, int const natoms 
, double const Z_core[] 
, int const echo=0 
); 

status_t all_tests(int const echo=0); 

} 
