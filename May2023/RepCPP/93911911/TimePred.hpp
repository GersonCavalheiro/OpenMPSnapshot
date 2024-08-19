

#ifndef EDGE_SEISMIC_KERNELS_TIME_PRED_HPP
#define EDGE_SEISMIC_KERNELS_TIME_PRED_HPP

#include "constants.hpp"
#include "data/Dynamic.h"
#include "dg/TimePred.hpp"

namespace edge {
namespace seismic {
namespace kernels {
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI,
unsigned short TL_N_CRS >
class TimePred;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI,
unsigned short TL_N_CRS >
class edge::seismic::kernels::TimePred {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

protected:
TL_T_REAL * m_rfs = nullptr;


static void storeStiffTDense( data::Dynamic         & io_dynMem,
TL_T_REAL             * o_stiffT[CE_MAX(TL_O_SP-1,1)][TL_N_DIS] ) {
dg::TimePred< TL_T_EL,
TL_O_SP,
TL_O_TI >::storeStiffTDense( io_dynMem,
o_stiffT );
}

public:

TimePred( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ) {
if( TL_N_RMS > 0 ) {
std::size_t l_size = TL_N_RMS * sizeof(TL_T_REAL);
m_rfs = (TL_T_REAL *) io_dynMem.allocate( l_size );

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_rfs[l_rm] = i_rfs[l_rm];
}
}
}


static void inline evalTimePrediction( unsigned short   i_nPts,
TL_T_REAL const *i_pts,
TL_T_REAL const  i_der[TL_O_TI][TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL       (*o_preDofs)[TL_N_QTS_E][TL_N_MDS][TL_N_CRS] ) {
for( unsigned short l_pt = 0; l_pt < i_nPts; l_pt++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
for( unsigned short l_ru = 0; l_ru < TL_N_CRS; l_ru++ ) o_preDofs[l_pt][l_qt][l_md][l_ru] = i_der[0][l_qt][l_md][l_ru];

TL_T_REAL l_scalar = 1.0;

for( unsigned short l_de = 1; l_de < TL_O_TI; l_de++ ) {
l_scalar *= i_pts[l_pt] / l_de;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
for( unsigned short l_md = 0; l_md < CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ); l_md++ ) {
for( unsigned short l_ru = 0; l_ru < TL_N_CRS; l_ru++ )
o_preDofs[l_pt][l_qt][l_md][l_ru] += l_scalar * i_der[l_de][l_qt][l_md][l_ru];
}
}
}
}
}


static void integrate( TL_T_REAL       i_dt,
TL_T_REAL const i_derE[TL_O_TI][TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL       o_tIntE[TL_N_QTS_E][TL_N_MDS][TL_N_CRS] ) {
TL_T_REAL l_sca = i_dt;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
#if PP_N_CRUNS==1
#pragma omp simd
#endif
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
#if PP_N_CRUNS>1
#pragma omp simd
#endif
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_tIntE[l_qt][l_md][l_cr] = l_sca * i_derE[0][l_qt][l_md][l_cr];

for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
l_sca *= i_dt / (l_de+1);

unsigned short l_nCpMds = (TL_N_RMS == 0) ? CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ) : TL_N_MDS;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
#if PP_N_CRUNS==1
#pragma omp simd
#endif
for( unsigned short l_md = 0; l_md < l_nCpMds; l_md++ )
#if PP_N_CRUNS>1
#pragma omp simd
#endif
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_tIntE[l_qt][l_md][l_cr] += l_sca * i_derE[l_de][l_qt][l_md][l_cr];
}
}

};

#endif
