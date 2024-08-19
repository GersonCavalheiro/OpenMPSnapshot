
#ifndef EDGE_SEISMIC_KERNELS_SURF_INT_SINGLE_HPP
#define EDGE_SEISMIC_KERNELS_SURF_INT_SINGLE_HPP

#include "SurfInt.hpp"
#include "dg/Basis.h"
#include "data/MmXsmmSingle.hpp"

namespace edge {
namespace seismic {
namespace kernels { 
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class SurfIntSingle;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class edge::seismic::kernels::SurfIntSingle: public edge::seismic::kernels::SurfInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
1 > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_MDS_FA = CE_N_ELEMENT_MODES( C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP );

static unsigned short const TL_N_MDS_EL = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_FMNS = CE_N_FLUXN_MATRICES( TL_T_EL );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_E = CE_N_ENS_FS_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_A = CE_N_ENS_FS_A_DE( TL_N_DIS );

TL_T_REAL *m_fIntLN[TL_N_FAS+TL_N_FMNS] = {};

TL_T_REAL *m_fIntT[TL_N_FAS] = {};

edge::data::MmXsmmSingle< TL_T_REAL > m_mm;


void generateKernels() {
m_mm.add( 0,                           
TL_N_MDS_FA,                 
TL_N_QTS_E,                  
TL_N_MDS_EL,                 
TL_N_MDS_FA,                 
TL_N_MDS_EL,                 
TL_N_MDS_FA,                 
static_cast<real_base>(1.0), 
static_cast<real_base>(0.0), 
LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD );

m_mm.add( 0,                           
TL_N_MDS_FA,                 
TL_N_QTS_E,                  
TL_N_QTS_E,                  
TL_N_MDS_FA,                 
TL_N_QTS_E,                  
TL_N_MDS_FA,                 
static_cast<real_base>(1.0), 
static_cast<real_base>(0.0), 
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 0,                           
TL_N_MDS_EL,                 
TL_N_QTS_E,                  
TL_N_MDS_FA,                 
TL_N_MDS_EL,                 
TL_N_MDS_FA,                 
TL_N_MDS_EL,                 
static_cast<real_base>(1.0), 
static_cast<real_base>(1.0), 
LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD );

m_mm.add( 1,                           
TL_N_MDS_FA,                 
TL_N_QTS_M,                  
TL_N_QTS_E,                  
TL_N_MDS_FA,                 
TL_N_QTS_E,                  
TL_N_MDS_FA,                 
static_cast<real_base>(1.0), 
static_cast<real_base>(0.0), 
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 1,                           
TL_N_MDS_EL,                 
TL_N_QTS_M,                  
TL_N_MDS_FA,                 
TL_N_MDS_EL,                 
TL_N_MDS_FA,                 
TL_N_MDS_EL,                 
static_cast<real_base>(1.0), 
static_cast<real_base>(1.0), 
LIBXSMM_GEMM_PREFETCH_NONE );
}

public:

SurfIntSingle( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): SurfInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
1 >( i_rfs,
io_dynMem ) {
this->storeFluxDense( io_dynMem,
m_fIntLN,
m_fIntT );

generateKernels();
}


void local( TL_T_REAL const   i_fsE[TL_N_FAS][TL_N_ENS_FS_E],
TL_T_REAL const (*i_fsA)[TL_N_ENS_FS_A],
TL_T_REAL const   i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
TL_T_REAL         io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
TL_T_REAL       (*io_dofsA)[TL_N_QTS_M][TL_N_MDS_EL][1],
TL_T_REAL         o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][1],
TL_T_REAL const   i_dofsP[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr,
TL_T_REAL const   i_tDofsP[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr ) const {
TL_T_REAL l_upAn[TL_N_QTS_M][TL_N_MDS_EL][1];
if( TL_N_RMS > 0 ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
#pragma omp simd
for( unsigned short l_md = 0; l_md < TL_N_MDS_EL; l_md++ ) {
l_upAn[l_qt][l_md][0] = 0;
}
}
}

for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
m_mm.m_kernels[0][0]( m_fIntLN[l_fa],
i_tDofsE[0][0],
o_scratch[0][0][0],
nullptr,
i_dofsP[0][0],
nullptr );

m_mm.m_kernels[0][1]( o_scratch[0][0][0],
i_fsE[l_fa],
o_scratch[1][0][0] );

m_mm.m_kernels[0][2]( m_fIntT[l_fa],
o_scratch[1][0][0],
io_dofsE[0][0],
nullptr,
i_tDofsP[0][0],
nullptr );

if( TL_N_RMS > 0 ) {
m_mm.m_kernels[1][0]( o_scratch[0][0][0],
i_fsA[l_fa],
o_scratch[1][0][0] );

m_mm.m_kernels[1][1]( m_fIntT[l_fa],
o_scratch[1][0][0],
l_upAn[0][0] );
}
}

if( TL_N_RMS > 0) this->scatterUpdateA( l_upAn, io_dofsA );
}


void neighFluxInt( unsigned short       i_fa,
unsigned short       i_vId,
unsigned short       i_fId,
TL_T_REAL      const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
TL_T_REAL            o_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][1],
TL_T_REAL      const i_pre[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr ) const {
unsigned short l_fMatId = std::numeric_limits< unsigned short >::max();
if( i_vId != std::numeric_limits< unsigned short >::max() ) {
l_fMatId = TL_N_FAS + this->fMatId( i_vId, i_fId );
}
else {
l_fMatId = i_fa;
}

m_mm.m_kernels[0][0](                     m_fIntLN[l_fMatId],
i_tDofsE[0][0],
o_tDofsFiE[0][0],
nullptr,
(TL_T_REAL const *) i_pre,
nullptr );
}


void neigh( unsigned short       i_fa,
unsigned short       i_vId,
unsigned short       i_fId,
TL_T_REAL      const i_fsE[TL_N_ENS_FS_E],
TL_T_REAL      const i_fsA[TL_N_ENS_FS_A],
TL_T_REAL      const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
TL_T_REAL      const i_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][1],
TL_T_REAL            io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
TL_T_REAL            io_dofsA[TL_N_QTS_M][TL_N_MDS_EL][1],
TL_T_REAL            o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][1],
TL_T_REAL      const i_pre[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr ) const {
TL_T_REAL const * l_tDofsFiE = o_scratch[0][0][0];

if( i_tDofsFiE == nullptr ) {
neighFluxInt( i_fa,
i_vId,
i_fId,
i_tDofsE,
o_scratch[0],
i_pre );
}
else {
l_tDofsFiE = i_tDofsFiE[0][0];
}

m_mm.m_kernels[0][1]( l_tDofsFiE,
i_fsE,
o_scratch[1][0][0] );

m_mm.m_kernels[0][2](                     m_fIntT[i_fa],
o_scratch[1][0][0],
io_dofsE[0][0],
nullptr,
(TL_T_REAL const *) i_pre,
nullptr );

if( TL_N_RMS > 0 ) {
m_mm.m_kernels[1][0]( l_tDofsFiE,
i_fsA,
o_scratch[1][0][0] );

m_mm.m_kernels[1][1]( m_fIntT[i_fa],
o_scratch[1][0][0],
io_dofsA[0][0] );
}
}
};

#endif