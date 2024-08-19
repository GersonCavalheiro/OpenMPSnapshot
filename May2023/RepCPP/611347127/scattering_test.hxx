#pragma once

#include "radial_grid.h" 
#include "energy_level.hxx" 
#include "status.hxx" 

namespace scattering_test {

status_t expand_sho_projectors( 
double prj[]  
, int const stride 
, radial_grid_t const & rg 
, double const sigma 
, int const numax 
, int const rpow=0 
, int const echo=0 
, double dprj[]=nullptr 
); 

status_t logarithmic_derivative(
radial_grid_t const rg[TRU_AND_SMT] 
, double const *const rV[TRU_AND_SMT] 
, double const sigma 
, int const ellmax 
, int const numax
, double const aHm[] 
, double const aSm[] 
, double const energy_range[3] 
, char const *label="" 
, int const echo=0 
, float const Rlog_over_sigma=6.f
); 

status_t eigenstate_analysis(
radial_grid_t const & gV 
, double const Vsmt[] 
, double const sigma 
, int const ellmax 
, int const numax 
, double const aHm[] 
, double const aSm[] 
, int const nr=384 
, double const Vshift=0 
, char const *label="" 
, int const echo=0 
, float const reference[3][4]=nullptr 
, float const warning_threshold=3e-3
); 

status_t emm_average(
double Mln[] 
, double const Mlmn[] 
, int const numax 
, int const avg2sum0=2 
, int const echo=0 
, int const stride=-1 
); 

status_t all_tests(int const echo=0); 

} 
