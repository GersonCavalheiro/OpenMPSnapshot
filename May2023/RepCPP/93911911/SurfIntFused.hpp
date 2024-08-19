
#ifndef EDGE_SEISMIC_KERNELS_SURF_INT_FUSED_HPP
#define EDGE_SEISMIC_KERNELS_SURF_INT_FUSED_HPP

#include "SurfInt.hpp"
#include "dg/Basis.h"
#include "data/MmXsmmFused.hpp"
#include "FakeMats.hpp"

#ifdef PP_MMKERNEL_PERF
#include "parallel/global.h"
#endif 

namespace edge {
namespace seismic {
namespace kernels { 
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class SurfIntFused;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class edge::seismic::kernels::SurfIntFused: public edge::seismic::kernels::SurfInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_N_CRS > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_FAS_DIV2 =  TL_N_FAS / 2;

static unsigned short const TL_N_MDS_FA = CE_N_ELEMENT_MODES( C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP );

static unsigned short const TL_N_MDS_EL = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_FMNS = CE_N_FLUXN_MATRICES( TL_T_EL );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_E = CE_N_ENS_FS_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_A = CE_N_ENS_FS_A_DE( TL_N_DIS );

TL_T_REAL *m_fIntLN[TL_N_FAS+TL_N_FMNS] = {};

TL_T_REAL *m_fIntT[TL_N_FAS] = {};

edge::data::MmXsmmFused< TL_T_REAL > m_mm;


static void getCscFlux( TL_T_REAL                const   i_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL                const   i_fIntN[TL_N_FMNS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL                const   i_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL],
std::vector< size_t >          & o_offsets,
std::vector< TL_T_REAL >       & o_nonZeros,
std::vector< t_matCsc >        & o_mats ) {
std::string l_cscFillIn = "none";

if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM ) {
l_cscFillIn = "qfma";
}

o_offsets.resize( 0 );
o_offsets.push_back( 0 );
o_nonZeros.resize( 0 );
o_mats.resize( 0 );

for( unsigned short l_fl = 0; l_fl < TL_N_FAS; l_fl++ ) {
t_matCsc l_fluxCsc;

edge::linalg::Matrix::denseToCsc< TL_T_REAL >( TL_N_MDS_EL,
TL_N_MDS_FA,
i_fIntL[l_fl][0],
l_fluxCsc,
TOL.BASIS,
std::numeric_limits< unsigned int >::max(),
std::numeric_limits< unsigned int >::max(),
l_cscFillIn );
o_mats.push_back( l_fluxCsc );

for( unsigned short l_nz = 0; l_nz < l_fluxCsc.val.size(); l_nz++ ) {
o_nonZeros.push_back( l_fluxCsc.val[l_nz] );
}
o_offsets.push_back( o_offsets.back() + l_fluxCsc.val.size() );
}

for( unsigned short l_fn = 0; l_fn < TL_N_FMNS; l_fn++ ) {
t_matCsc l_fluxCsc;

edge::linalg::Matrix::denseToCsc< TL_T_REAL >( TL_N_MDS_EL,
TL_N_MDS_FA,
i_fIntN[l_fn][0],
l_fluxCsc,
TOL.BASIS,
std::numeric_limits< unsigned int >::max(),
std::numeric_limits< unsigned int >::max(),
l_cscFillIn );
o_mats.push_back( l_fluxCsc );

for( unsigned short l_nz = 0; l_nz < l_fluxCsc.val.size(); l_nz++ ) {
o_nonZeros.push_back( l_fluxCsc.val[l_nz] );
}
o_offsets.push_back( o_offsets.back() + l_fluxCsc.val.size() );
}

for( unsigned short l_ft = 0; l_ft < TL_N_FAS; l_ft++ ) {
t_matCsc l_fluxCsc;

edge::linalg::Matrix::denseToCsc< TL_T_REAL >( TL_N_MDS_FA,
TL_N_MDS_EL,
i_fIntT[l_ft][0],
l_fluxCsc,
TOL.BASIS,
std::numeric_limits< unsigned int >::max(),
std::numeric_limits< unsigned int >::max(),
l_cscFillIn );
o_mats.push_back( l_fluxCsc );

for( unsigned short l_nz = 0; l_nz < l_fluxCsc.val.size(); l_nz++ ) {
o_nonZeros.push_back( l_fluxCsc.val[l_nz] );
}
o_offsets.push_back( o_offsets.back() + l_fluxCsc.val.size() );
}
}


void generateKernels( TL_T_REAL const   i_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL const   i_fIntN[TL_N_FMNS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL const   i_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL] ) {
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_fIntCsc;
getCscFlux( i_fIntL,
i_fIntN,
i_fIntT,
l_offsets,
l_nonZeros,
l_fIntCsc );

for( unsigned short l_fl = 0; l_fl < TL_N_FAS; l_fl++ ) {
m_mm.add(  0,                         
TL_N_CRS,                  
false,                     
&l_fIntCsc[l_fl].colPtr[0], 
&l_fIntCsc[l_fl].rowIdx[0], 
&l_fIntCsc[l_fl].val[0],    
TL_N_QTS_E,                
TL_N_MDS_FA,               
TL_N_MDS_EL,               
TL_N_MDS_EL,               
0,                         
TL_N_MDS_FA,               
TL_T_REAL(1.0),            
TL_T_REAL(0.0),            
LIBXSMM_GEMM_PREFETCH_NONE );
}

for( unsigned short l_fn = 0; l_fn < TL_N_FMNS; l_fn++ ) {
unsigned short l_ma = TL_N_FAS + l_fn;

m_mm.add(  0,                         
TL_N_CRS,                  
false,                     
&l_fIntCsc[l_ma].colPtr[0], 
&l_fIntCsc[l_ma].rowIdx[0], 
&l_fIntCsc[l_ma].val[0],    
TL_N_QTS_E,                
TL_N_MDS_FA,               
TL_N_MDS_EL,               
TL_N_MDS_EL,               
0,                         
TL_N_MDS_FA,               
TL_T_REAL(1.0),            
TL_T_REAL(0.0),            
LIBXSMM_GEMM_PREFETCH_NONE );
}

for( unsigned short l_ft = 0; l_ft < TL_N_FAS; l_ft++ ) {
unsigned short l_ma = TL_N_FAS + TL_N_FMNS + l_ft;

m_mm.add(  2,                         
TL_N_CRS,                  
false,                     
&l_fIntCsc[l_ma].colPtr[0], 
&l_fIntCsc[l_ma].rowIdx[0], 
&l_fIntCsc[l_ma].val[0],    
TL_N_QTS_E,                
TL_N_MDS_EL,               
TL_N_MDS_FA,               
TL_N_MDS_FA,               
0,                         
TL_N_MDS_EL,               
TL_T_REAL(1.0),            
TL_T_REAL(1.0),            
LIBXSMM_GEMM_PREFETCH_NONE );
}

for( unsigned short l_ft = 0; l_ft < TL_N_FAS; l_ft++ ) {
unsigned short l_ma = TL_N_FAS + TL_N_FMNS + l_ft;

m_mm.add(  4,                         
TL_N_CRS,                  
false,                     
&l_fIntCsc[l_ma].colPtr[0], 
&l_fIntCsc[l_ma].rowIdx[0], 
&l_fIntCsc[l_ma].val[0],    
TL_N_QTS_M,                
TL_N_MDS_EL,               
TL_N_MDS_FA,               
TL_N_MDS_FA,               
0,                         
TL_N_MDS_EL,               
TL_T_REAL(1.0),            
TL_T_REAL(1.0),            
LIBXSMM_GEMM_PREFETCH_NONE );
}

t_matCsr l_fsCsrE;
FakeMats< TL_N_DIS >::fsCsrE( l_fsCsrE );
EDGE_CHECK_EQ( l_fsCsrE.val.size(), TL_N_QTS_E*TL_N_QTS_E );

m_mm.add(  1,                   
TL_N_CRS,            
true,                
&l_fsCsrE.rowPtr[0], 
&l_fsCsrE.colIdx[0], 
&l_fsCsrE.val[0],    
TL_N_QTS_E,          
TL_N_MDS_FA,         
TL_N_QTS_E,          
0,                   
TL_N_MDS_FA,         
TL_N_MDS_FA,         
TL_T_REAL(1.0),      
TL_T_REAL(0.0),      
LIBXSMM_GEMM_PREFETCH_BL2_VIA_C );

if( TL_N_RMS > 0 ) {
t_matCsr l_fsCsrA;
FakeMats< TL_N_DIS >::fsCsrA( l_fsCsrA );
EDGE_CHECK_EQ( l_fsCsrA.val.size(), TL_N_QTS_M*TL_N_QTS_E );

m_mm.add(  3,                   
TL_N_CRS,            
true,                
&l_fsCsrA.rowPtr[0], 
&l_fsCsrA.colIdx[0], 
&l_fsCsrA.val[0],    
TL_N_QTS_M,          
TL_N_MDS_FA,         
TL_N_QTS_M,          
0,                   
TL_N_MDS_FA,         
TL_N_MDS_FA,         
TL_T_REAL(1.0),      
TL_T_REAL(0.0),      
LIBXSMM_GEMM_PREFETCH_NONE );
}
}


static void storeFluxSparse( TL_T_REAL     const   i_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL     const   i_fIntN[TL_N_FMNS][TL_N_MDS_EL][TL_N_MDS_FA],
TL_T_REAL     const   i_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL],
data::Dynamic       & io_dynMem,
TL_T_REAL           * o_fIntLN[TL_N_FAS+TL_N_FMNS],
TL_T_REAL           * o_fIntT[TL_N_FAS] ) {
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_fIntCsc;
getCscFlux( i_fIntL,
i_fIntN,
i_fIntT,
l_offsets,
l_nonZeros,
l_fIntCsc );

TL_T_REAL * l_fIntRaw = (TL_T_REAL*) io_dynMem.allocate( l_nonZeros.size() * sizeof(TL_T_REAL),
4096,
true );
for( std::size_t l_en = 0; l_en < l_nonZeros.size(); l_en++ ) {
l_fIntRaw[l_en] = l_nonZeros[l_en];
}

EDGE_CHECK_EQ( l_offsets.size(), TL_N_FAS + TL_N_FMNS + TL_N_FAS + 1 );

for( unsigned short l_ma = 0; l_ma < TL_N_FAS+TL_N_FMNS; l_ma++ ) {
o_fIntLN[l_ma] = l_fIntRaw + l_offsets[l_ma];
}
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
o_fIntT[l_fa] = l_fIntRaw + l_offsets[TL_N_FAS+TL_N_FMNS + l_fa];
}
}

inline void execMmKernel( unsigned short         i_group, 
unsigned short         i_kernel,
TL_T_REAL      const * i_a,
TL_T_REAL      const * i_b,
TL_T_REAL            * io_c ) {
#ifdef PP_MMKERNEL_PERF
size_t start = _rdtsc();
#endif
m_mm.m_kernels[i_group][i_kernel]( i_a, i_b, io_c );
#ifdef PP_MMKERNEL_PERF
size_t duration = _rdtsc() - start;
m_mm.m_kernelStats[edge::parallel::g_thread][i_group][i_kernel].invocations++;
m_mm.m_kernelStats[edge::parallel::g_thread][i_group][i_kernel].cycles += duration; 
#endif
}

inline void execMmKernelPf( unsigned short         i_group, 
unsigned short         i_kernel,
TL_T_REAL      const * i_a,
TL_T_REAL      const * i_b,
TL_T_REAL            * io_c,
TL_T_REAL      const * i_a_pf,
TL_T_REAL      const * i_b_pf,
TL_T_REAL      const * i_c_pf  ) {
#ifdef PP_MMKERNEL_PERF
size_t start = _rdtsc();
#endif
m_mm.m_kernels[i_group][i_kernel]( i_a, i_b, io_c, i_a_pf, i_b_pf, i_c_pf );
#ifdef PP_MMKERNEL_PERF
size_t duration = _rdtsc() - start;
m_mm.m_kernelStats[edge::parallel::g_thread][i_group][i_kernel].invocations++;
m_mm.m_kernelStats[edge::parallel::g_thread][i_group][i_kernel].cycles += duration; 
#endif
}

public:

SurfIntFused( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): SurfInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_N_CRS >( i_rfs,
io_dynMem ) {
dg::Basis l_basis( TL_T_EL,
TL_O_SP );

TL_T_REAL l_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA];
TL_T_REAL l_fIntN[TL_N_FMNS][TL_N_MDS_EL][TL_N_MDS_FA];
TL_T_REAL l_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL];
l_basis.getFluxDense( l_fIntL[0][0],
l_fIntN[0][0],
l_fIntT[0][0] );

storeFluxSparse( l_fIntL,
l_fIntN,
l_fIntT,
io_dynMem,
m_fIntLN,
m_fIntT );

generateKernels( l_fIntL,
l_fIntN,
l_fIntT );
}


void local( TL_T_REAL const   i_fsE[TL_N_FAS][TL_N_ENS_FS_E],
TL_T_REAL const (*i_fsA)[TL_N_ENS_FS_A],
TL_T_REAL const   i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL         io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL       (*io_dofsA)[TL_N_QTS_M][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL         o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][TL_N_CRS],
TL_T_REAL const   i_dofsP[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS] = nullptr,
TL_T_REAL const   i_tDofsP[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS] = nullptr ) {
TL_T_REAL l_upAn[TL_N_QTS_M][TL_N_MDS_EL][TL_N_CRS];
if( TL_N_RMS > 0 ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS_EL; l_md++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
l_upAn[l_qt][l_md][l_cr] = 0;
}
}
}
}

for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
execMmKernel( 0, l_fa, i_tDofsE[0][0],
m_fIntLN[l_fa],
o_scratch[0][0][0] );

execMmKernelPf( 1, 0, i_fsE[l_fa],
o_scratch[0][0][0],
o_scratch[1][0][0],
nullptr,
(l_fa < TL_N_FAS_DIV2) ? i_dofsP[0][0] : i_tDofsP[0][0],
nullptr );

execMmKernel( 2, l_fa, o_scratch[1][0][0],
m_fIntT[l_fa],
io_dofsE[0][0] );

if( TL_N_RMS > 0 ) {
execMmKernel( 3, 0, i_fsA[l_fa],
o_scratch[0][0][0],
o_scratch[1][0][0] );

execMmKernel( 4, l_fa, o_scratch[1][0][0],
m_fIntT[l_fa],
l_upAn[0][0] );
}
}

if( TL_N_RMS > 0) this->scatterUpdateA( l_upAn, io_dofsA );
}


void neighFluxInt( unsigned short       i_fa,
unsigned short       i_vId,
unsigned short       i_fId,
TL_T_REAL      const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL            o_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][TL_N_CRS],
TL_T_REAL      const i_pre[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS] = nullptr ) {
unsigned short l_fMatId = std::numeric_limits< unsigned short >::max();
if( i_vId != std::numeric_limits< unsigned short >::max() ) {
l_fMatId = TL_N_FAS + this->fMatId( i_vId, i_fId );
}
else {
l_fMatId = i_fa;
}

execMmKernel( 0, l_fMatId, i_tDofsE[0][0],
m_fIntLN[l_fMatId],
o_tDofsFiE[0][0] );
}


void neigh( unsigned short       i_fa,
unsigned short       i_vId,
unsigned short       i_fId,
TL_T_REAL      const i_fsE[TL_N_ENS_FS_E],
TL_T_REAL      const i_fsA[TL_N_ENS_FS_A],
TL_T_REAL      const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL      const i_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][TL_N_CRS],
TL_T_REAL            io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL            io_dofsA[TL_N_QTS_M][TL_N_MDS_EL][TL_N_CRS],
TL_T_REAL            o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][TL_N_CRS],
TL_T_REAL      const i_pre[TL_N_QTS_E][TL_N_MDS_EL][TL_N_CRS] = nullptr ) {
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

execMmKernelPf( 1, 0,                     i_fsE,
l_tDofsFiE,
o_scratch[1][0][0],
nullptr,
(TL_T_REAL const *) i_pre,
nullptr );

execMmKernel( 2, i_fa, o_scratch[1][0][0],
m_fIntT[i_fa],
io_dofsE[0][0] );

if( TL_N_RMS > 0 ) {
execMmKernel( 3, 0, i_fsA,
l_tDofsFiE,
o_scratch[1][0][0] );

execMmKernel( 4, i_fa, o_scratch[1][0][0],
m_fIntT[i_fa],
io_dofsA[0][0] );
}
}
};

#endif
