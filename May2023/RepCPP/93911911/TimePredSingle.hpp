
#ifndef EDGE_SEISMIC_KERNELS_TIME_PRED_SINGLE_HPP
#define EDGE_SEISMIC_KERNELS_TIME_PRED_SINGLE_HPP

#include "TimePred.hpp"
#include "data/MmXsmmSingle.hpp"

namespace edge {
namespace seismic {
namespace kernels {
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI >
class TimePredSingle;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI >
class edge::seismic::kernels::TimePredSingle: public edge::seismic::kernels::TimePred < TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_O_TI,
1 > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_ENS_STAR_E = CE_N_ENS_STAR_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_STAR_A = CE_N_ENS_STAR_A_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_SRC_A = CE_N_ENS_SRC_A_DE( TL_N_DIS );

TL_T_REAL *m_stiffT[CE_MAX(TL_O_TI-1,1)][TL_N_DIS] = {};

edge::data::MmXsmmSingle< TL_T_REAL > m_mm;


void generateKernels() {
for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
m_mm.add( 0,                                                 
CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ),   
TL_N_QTS_E,                                        
CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de-1 ), 
TL_N_MDS,                                          
TL_N_MDS,                                          
TL_N_MDS,                                          
static_cast<TL_T_REAL>(1.0),                       
static_cast<TL_T_REAL>(0.0),                       
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 1,                                               
CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ), 
TL_N_QTS_E,                                      
TL_N_QTS_E,                                      
TL_N_MDS,                                        
TL_N_QTS_E,                                      
TL_N_MDS,                                        
static_cast<TL_T_REAL>(1.0),                     
static_cast<TL_T_REAL>(1.0),                     
LIBXSMM_GEMM_PREFETCH_NONE );

if( TL_N_RMS > 0 ) {
m_mm.add( 2,                                            
CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, 1 ), 
TL_N_QTS_M,                                   
TL_N_DIS,                                     
TL_N_MDS,                                     
TL_N_DIS,                                     
TL_N_MDS,                                     
static_cast<TL_T_REAL>(1.0),                  
static_cast<TL_T_REAL>(1.0),                  
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 2,                           
TL_N_MDS,                    
TL_N_QTS_M,                  
TL_N_QTS_M,                  
TL_N_MDS,                    
TL_N_QTS_M,                  
TL_N_MDS,                    
static_cast<TL_T_REAL>(1.0), 
static_cast<TL_T_REAL>(1.0), 
LIBXSMM_GEMM_PREFETCH_NONE );
}
}
}

public:

TimePredSingle( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): TimePred< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_O_TI,
1 >( i_rfs,
io_dynMem ) {
this->storeStiffTDense( io_dynMem,
m_stiffT );

generateKernels();
};


void ck( TL_T_REAL         i_dT,
TL_T_REAL const   i_starE[TL_N_DIS][TL_N_ENS_STAR_E],
TL_T_REAL const (*i_starA)[TL_N_ENS_STAR_A],
TL_T_REAL const (*i_srcA)[TL_N_ENS_SRC_A],
TL_T_REAL const   i_dofsE[TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL const (*i_dofsA)[TL_N_QTS_M][TL_N_MDS][1],
TL_T_REAL         o_scratch[TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL         o_derE[TL_O_TI][TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL       (*o_derA)[TL_O_TI][TL_N_QTS_M][TL_N_MDS][1],
TL_T_REAL         o_tIntE[TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL       (*o_tIntA)[TL_N_QTS_M][TL_N_MDS][1] ) const {
TL_T_REAL const *l_rfs = this->m_rfs;

TL_T_REAL l_scalar = i_dT;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_derE[0][l_qt][l_md][0] = i_dofsE[l_qt][l_md][0];
o_tIntE[l_qt][l_md][0]   = l_scalar * i_dofsE[l_qt][l_md][0];
}
}

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_derA[l_rm][0][l_qt][l_md][0] = i_dofsA[l_rm][l_qt][l_md][0];
o_tIntA[l_rm][l_qt][l_md][0] = l_scalar * i_dofsA[l_rm][l_qt][l_md][0];
}
}
}

for( unsigned short l_de = 1; l_de < TL_O_TI; l_de++ ) {
unsigned short l_re = (TL_N_RMS == 0) ? l_de : 1;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) o_derE[l_de][l_qt][l_md][0] = 0;

TL_T_REAL l_scratch[TL_N_QTS_M][TL_N_MDS];
if( TL_N_RMS > 0 ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) l_scratch[l_qt][l_md] = 0;
}
}

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
m_mm.m_kernels[0][l_re-1]( m_stiffT[l_re-1][l_di],
o_derE[l_de-1][0][0],
o_scratch[0][0] );
m_mm.m_kernels[1][l_re-1]( o_scratch[0][0],
i_starE[l_di],
o_derE[l_de][0][0] );

if( TL_N_RMS > 0 ) {
m_mm.m_kernels[2][0]( o_scratch[TL_N_QTS_M][0],
i_starA[l_di],
l_scratch[0] );
}
}

l_scalar *= i_dT / (l_de+1);

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_mm.m_kernels[2][1]( o_derA[l_rm][l_de-1][0][0],
i_srcA[l_rm],
o_derE[l_de][0][0] );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_derA[l_rm][l_de][l_qt][l_md][0] = l_rfs[l_rm] * ( l_scratch[l_qt][l_md]+ o_derA[l_rm][l_de-1][l_qt][l_md][0] );
o_tIntA[l_rm][l_qt][l_md][0] += l_scalar * o_derA[l_rm][l_de][l_qt][l_md][0];
}
}
}

unsigned short l_nCpMds = (TL_N_RMS == 0) ? CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ) : TL_N_MDS;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < l_nCpMds; l_md++ ) {
o_tIntE[l_qt][l_md][0] += l_scalar * o_derE[l_de][l_qt][l_md][0];
}
}
}
}
};

#endif