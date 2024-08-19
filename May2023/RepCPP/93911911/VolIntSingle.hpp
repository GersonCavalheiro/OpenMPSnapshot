
#ifndef EDGE_SEISMIC_KERNELS_VOL_INT_SINGLE_HPP
#define EDGE_SEISMIC_KERNELS_VOL_INT_SINGLE_HPP

#include "VolInt.hpp"
#include "dg/Basis.h"
#include "data/MmXsmmSingle.hpp"

namespace edge {
namespace seismic {
namespace kernels { 
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class VolIntSingle;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class edge::seismic::kernels::VolIntSingle: edge::seismic::kernels::VolInt < TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
1 > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_ENS_STAR_E = CE_N_ENS_STAR_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_STAR_A = CE_N_ENS_STAR_A_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_SRC_A = CE_N_ENS_SRC_A_DE( TL_N_DIS );

edge::data::MmXsmmSingle< TL_T_REAL > m_mm;

TL_T_REAL *m_stiff[TL_N_DIS] = {};


void generateKernels() {
m_mm.add( 0,                                            
TL_N_MDS,                                     
TL_N_QTS_E,                                   
CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, 1 ), 
TL_N_MDS,                                     
TL_N_MDS,                                     
TL_N_MDS,                                     
static_cast<TL_T_REAL>(1.0),                  
static_cast<TL_T_REAL>(0.0),                  
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 0,                           
TL_N_MDS,                    
TL_N_QTS_E,                  
TL_N_QTS_E,                  
TL_N_MDS,                    
TL_N_QTS_E,                  
TL_N_MDS,                    
static_cast<TL_T_REAL>(1.0), 
static_cast<TL_T_REAL>(1.0), 
LIBXSMM_GEMM_PREFETCH_NONE );

if( TL_N_RMS > 0 ) {
m_mm.add( 1,                           
TL_N_MDS,                    
TL_N_QTS_M,                  
TL_N_DIS,                    
TL_N_MDS,                    
TL_N_DIS,                    
TL_N_MDS,                    
static_cast<TL_T_REAL>(1.0), 
static_cast<TL_T_REAL>(1.0), 
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 1,                           
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

public:

VolIntSingle( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): VolInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
1 >( i_rfs,
io_dynMem ) {
this->storeStiffDense( io_dynMem,
m_stiff );

generateKernels();
}


void apply( TL_T_REAL const   i_starE[TL_N_DIS][TL_N_ENS_STAR_E],
TL_T_REAL const (*i_starA)[TL_N_ENS_STAR_A],
TL_T_REAL const (*i_srcA)[TL_N_ENS_SRC_A],
TL_T_REAL const   i_tDofsE[TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL const (*i_tDofsA)[TL_N_QTS_M][TL_N_MDS][1],
TL_T_REAL         io_dofsE[TL_N_QTS_E][TL_N_MDS][1],
TL_T_REAL       (*io_dofsA)[TL_N_QTS_M][TL_N_MDS][1],
TL_T_REAL         o_scratch[TL_N_QTS_E][TL_N_MDS][1] ) const {
TL_T_REAL const *l_rfs = this->m_rfs;

TL_T_REAL l_scratch[TL_N_QTS_M][TL_N_MDS][1];
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ )
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) l_scratch[l_qt][l_md][0] = 0;

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
m_mm.m_kernels[0][0]( m_stiff[l_di],
i_tDofsE[0][0],
o_scratch[0][0] );

m_mm.m_kernels[0][1]( o_scratch[0][0],
i_starE[l_di],
io_dofsE[0][0] );

if( TL_N_RMS > 0 ) {
m_mm.m_kernels[1][0]( o_scratch[TL_N_QTS_M][0],
i_starA[l_di],
l_scratch[0][0] );
}
}

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_mm.m_kernels[1][1]( i_tDofsA[l_rm][0][0],
i_srcA[l_rm],
io_dofsE[0][0] );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
io_dofsA[l_rm][l_qt][l_md][0] += l_rfs[l_rm] * ( l_scratch[l_qt][l_md][0] - i_tDofsA[l_rm][l_qt][l_md][0] );
}
}
}
}
};

#endif