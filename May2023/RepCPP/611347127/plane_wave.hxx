#pragma once

#include <cstdio> 
#include <cstdint> 
#include <complex> 
#include <vector> 

#include "status.hxx" 
#include "data_view.hxx" 
#include "real_space.hxx" 
#include "inline_math.hxx" 

namespace plane_wave {

class DensityIngredients {
public:

view2D<std::complex<double>> psi_r; 
view2D<std::complex<double>> coeff; 
std::vector<double> energies; 
std::vector<uint32_t> offset; 
double kpoint_weight;
int kpoint_index;
int ng[3];  
int ncoeff; 
int nbands;
int natoms;
char tag[32];


void constructor( 
int const nG[3] 
, int const nBands=0
, int const nAtoms=0
, int const nCoeff=0
, double const k_weight=1.
, int const kpoint_id=-1
, int const echo=0 
) {
kpoint_weight = std::max(0.0, k_weight);
ncoeff = nCoeff;
nbands = nBands;
natoms = nAtoms;
kpoint_index = kpoint_id;
set(ng, 3, nG);
auto const nG_all = size_t(nG[2])*size_t(nG[1])*size_t(nG[0]);
auto const nG_aligned = align<0>(nG_all);
int  const nC_aligned = align<0>(ncoeff);
if (echo > 8) std::printf("# DensityIngredients allocates %.3f MByte for wave functions"
" + %.3f kByte for coefficients\n", nbands*16e-6*nG_aligned, nbands*16e-3*nC_aligned);
std::complex<double> const zero(0);
psi_r = view2D<std::complex<double>>(nbands, nG_aligned, zero);
coeff = view2D<std::complex<double>>(nbands, nC_aligned, zero);
std::snprintf(tag, 31, "kpoint #%i weight %g", kpoint_id, kpoint_weight);
} 

}; 


status_t solve(
int const natoms_prj 
, view2D<double> const & xyzZ 
, real_space::grid_t const & g 
, double const *const vtot 
, double const *const sigma_prj=nullptr
, int    const *const numax_prj=nullptr
, double *const *const atom_mat=nullptr 
, int const echo=0 
, std::vector<DensityIngredients> *export_rho=nullptr        
); 

status_t all_tests(int const echo=0); 

} 
