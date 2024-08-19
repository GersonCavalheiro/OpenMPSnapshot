
#ifndef EDGE_SEISMIC_KERNELS_SURF_INT_HPP
#define EDGE_SEISMIC_KERNELS_SURF_INT_HPP

#include "constants.hpp"
#include "data/Dynamic.h"
#include "dg/SurfInt.hpp"

namespace edge {
namespace seismic {
namespace kernels {
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class SurfInt;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class edge::seismic::kernels::SurfInt {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_MDS_FA = CE_N_ELEMENT_MODES( C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP );

static unsigned short const TL_N_MDS_EL = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_FMNS = CE_N_FLUXN_MATRICES( TL_T_EL );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

TL_T_REAL * m_rfs = nullptr;

protected:

static void storeFluxDense( data::Dynamic       & io_dynMem,
TL_T_REAL           * o_fIntLN[TL_N_FAS+TL_N_FMNS],
TL_T_REAL           * o_fIntT[TL_N_FAS] ) {
dg::SurfInt< TL_T_EL,
TL_O_SP >::storeFluxDense( io_dynMem,
o_fIntLN,
o_fIntT );
}


SurfInt( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ) {
if( TL_N_RMS > 0 ) {
std::size_t l_size = TL_N_RMS * sizeof(TL_T_REAL);
m_rfs = (TL_T_REAL *) io_dynMem.allocate( l_size );

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_rfs[l_rm] = i_rfs[l_rm];
}
}
}


static unsigned short inline fMatId( unsigned short i_vIdElFaEl,
unsigned short i_fIdElFaEl ) {
return i_vIdElFaEl*TL_N_FAS+i_fIdElFaEl;
}

public:

void scatterUpdateA( TL_T_REAL const   i_update[TL_N_QTS_M][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL       (*io_dofsA)[TL_N_QTS_M][TL_N_MDS_EL][TL_N_CRS] ) const {
for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#if PP_N_CRUNS==1
#pragma omp simd
#endif
for( unsigned short l_md = 0; l_md < TL_N_MDS_EL; l_md++ ) {
#if PP_N_CRUNS>1
#pragma omp simd
#endif
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ){
io_dofsA[l_rm][l_qt][l_md][l_cr] += m_rfs[l_rm] * i_update[l_qt][l_md][l_cr];
}
}
}
}
}
};

#endif
