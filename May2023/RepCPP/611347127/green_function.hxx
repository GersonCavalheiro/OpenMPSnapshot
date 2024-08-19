#pragma once

#include <cstdint>    
#include <vector>     
#include <complex>    

#include "status.hxx" 
#include "green_action.hxx" 
#include "green_dyadic.hxx" 




namespace green_function {

char const boundary_condition_name[][16] = {"isolated", "periodic", "vacuum"};
char const boundary_condition_shortname[][8] = {"iso", "peri", "vacu"};

status_t construct_Green_function(
green_action::plan_t & p 
, uint32_t const ng[3] 
, int8_t const boundary_condition[3] 
, double const hg[3] 
, std::vector<double> const & Veff 
, std::vector<double> const & xyzZinso 
, std::vector<std::vector<double>> const & AtomMatrices 
, int const echo=0 
, std::complex<double> const *energy_parameter=nullptr 
, int const Noco=2
); 

status_t update_atom_matrices(
green_dyadic::dyadic_plan_t & p
, std::complex<double> E_param
, std::vector<std::vector<double>> const & AtomMatrices
, double const dVol 
, int const Noco 
, double const scale_H 
, int const echo 
); 

status_t update_phases(
green_action::plan_t & p
, double const k_point[3]
, int const Noco=1
, int const echo=0 
); 

inline status_t update_energy_parameter(
green_action::plan_t & p
, std::complex<double> E_param
, std::vector<std::vector<double>> const & AtomMatrices
, double const dVol
, int const Noco=1
, double const scale_H=1
, int const echo=0
) {
p.E_param = E_param;
return update_atom_matrices(p.dyadic_plan, E_param, AtomMatrices, dVol, Noco, scale_H, echo);
} 

status_t all_tests(int const echo=0); 

} 
