#pragma once

#include <cstdio> 
#include <cmath> 
#include <algorithm> 

#include "energy_level.hxx" 
#include "display_units.h" 
#include "radial_grid.hxx" 

typedef struct energy_level_t<TRU_ONLY> spherical_state_t; 

int constexpr core=0, semicore=1, valence=2, csv_undefined=3; 

inline char const * csv_name(int const csv) { 
return (core     == csv) ? "core" : (
(valence  == csv) ? "valence" : (
(semicore == csv) ? "semicore" : "?" ) );
} 

inline void show_state(
char const *label 
, char const *csv_class 
, char const *tag 
, double const occ 
, double const energy 
, char const final='\n' 
) {
std::printf("# %s %-9s%-4s%6.1f E=%16.6f %s%c", label, csv_class, tag, occ, energy*eV, _eV, final);
} 

inline double show_state_analysis( 
int const echo 
, char const *label 
, radial_grid_t const & rg 
, double const wave[] 
, char const *tag 
, double const occ 
, double const energy 
, char const *csv_class 
, int const ir_cut=0 
) {

double q{0}, qr{0}, qr2{0}, qrm1{0}, qout{0};
for (int ir = 0; ir < rg.n; ++ir) {
double const rho_wf = wave[ir]*wave[ir];
double const dV = rg.r2dr[ir];
double const r = rg.r[ir];
double const r_inv_dV = rg.rdr[ir]; 
q    += rho_wf*dV; 
qr   += rho_wf*r*dV; 
qr2  += rho_wf*r*r*dV; 
qrm1 += rho_wf*r_inv_dV; 
qout += rho_wf*dV*(ir >= ir_cut); 
} 
double const qinv = (q > 0) ? 1./q : 0;
double const charge_outside = qout*qinv;

if (echo > 0) {
show_state(label, csv_class, tag, occ, energy, ' ');
if (echo > 4) { 
auto const rms = std::sqrt(std::max(0., qr2*qinv));
std::printf(" <r>=%g rms=%g %s <r^-1>=%g %s\t", qr*qinv*Ang, rms*Ang, _Ang, qrm1*qinv*eV, _eV);
} 
std::printf((charge_outside > 1e-3) ? "out=%6.2f %%\n"
: "out= %.0e %%\n", 100*charge_outside);
} 

return charge_outside; 
} 


