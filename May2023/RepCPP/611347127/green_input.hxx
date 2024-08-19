#pragma once

#include <vector> 

#include "status.hxx" 

namespace green_input {

status_t load_Hamiltonian(
uint32_t ng[3] 
, int8_t bc[3] 
, double hg[3] 
, std::vector<double> & Veff
, int & natoms
, std::vector<double> & xyzZinso
, std::vector<std::vector<double>> & atom_mat
, char const *const filename="Hmt.json" 
, int const echo=0 
); 

status_t all_tests(int echo=0); 

} 
