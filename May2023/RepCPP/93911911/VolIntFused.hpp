
#ifndef EDGE_SEISMIC_KERNELS_VOL_INT_FUSED_HPP
#define EDGE_SEISMIC_KERNELS_VOL_INT_FUSED_HPP

#include "VolInt.hpp"
#include "dg/Basis.h"
#include "data/MmXsmmFused.hpp"
#include "FakeMats.hpp"

namespace edge {
namespace seismic {
namespace kernels { 
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class VolIntFused;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class edge::seismic::kernels::VolIntFused: edge::seismic::kernels::VolInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_N_CRS > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_NZS_STAR_E = CE_N_ENS_STAR_E_SP( TL_N_DIS );

static unsigned short const TL_N_NZS_STAR_A = CE_N_ENS_STAR_A_SP( TL_N_DIS );

static unsigned short const TL_N_NZS_SRC_A = CE_N_ENS_SRC_A_SP( TL_N_DIS );

edge::data::MmXsmmFused< TL_T_REAL > m_mm;

TL_T_REAL *m_stiff[TL_N_DIS] = {};


static void getCscStiff( TL_T_REAL               const   i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS],
unsigned int                  & o_maxNzRow,
std::vector< size_t >         & o_offsets,
std::vector< TL_T_REAL >      & o_nonZeros,
std::vector< t_matCsc >       & o_mats ) {
std::string l_cscFillIn = "none";

if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM ) {
l_cscFillIn = "qfma";
}

o_maxNzRow = 0;
o_offsets.resize( 0 );
o_offsets.push_back( 0 );
o_nonZeros.resize( 0 );
o_mats.resize( 0 );

t_matCrd l_stiffCrd[TL_N_DIS];
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
edge::linalg::Matrix::denseToCrd< TL_T_REAL >( TL_N_MDS,
TL_N_MDS,
i_stiff[l_di][0],
l_stiffCrd[l_di],
TOL.BASIS );
}

unsigned int l_nzBl[2][2][2];
l_nzBl[0][0][0] = l_nzBl[0][1][0] = 0;
l_nzBl[0][0][1] = l_nzBl[0][1][1] = TL_N_MDS-1;

for( unsigned short l_di = 0; l_di < N_DIM; l_di++ ) {
edge::linalg::Matrix::getBlockNz( l_stiffCrd[l_di], l_nzBl[0], l_nzBl[1] );
o_maxNzRow = std::max( o_maxNzRow, l_nzBl[1][0][1] );
}

#ifdef PP_T_BASIS_HIERARCHICAL
EDGE_CHECK_EQ( o_maxNzRow+1, CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP-1 ) );
#endif

t_matCsc l_stiffCsc[TL_N_DIS];
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
edge::linalg::Matrix::denseToCsc< TL_T_REAL >( TL_N_MDS,
TL_N_MDS,
i_stiff[l_di][0],
l_stiffCsc[l_di],
TOL.BASIS,
std::numeric_limits< unsigned int >::max(),
std::numeric_limits< unsigned int >::max(),
l_cscFillIn );
o_mats.push_back( l_stiffCsc[l_di] );
}

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
for( unsigned int l_nz = 0; l_nz < l_stiffCsc[l_di].val.size(); l_nz++ ) {
o_nonZeros.push_back( l_stiffCsc[l_di].val[l_nz] );
}
o_offsets.push_back( o_offsets.back() + l_stiffCsc[l_di].val.size() );
}
}


void generateKernels( TL_T_REAL const i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS] ) {
unsigned int l_maxNzRow;
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_stiffCsc;
getCscStiff( i_stiff,
l_maxNzRow,
l_offsets,
l_nonZeros,
l_stiffCsc );

t_matCsr l_starCsrE;
FakeMats< TL_N_DIS >::starCsrE( l_starCsrE );
EDGE_CHECK_EQ( l_starCsrE.val.size(), TL_N_NZS_STAR_E );

for( unsigned short l_di = 0; l_di < N_DIM; l_di++ ) {
m_mm.add(  0,                                                 
TL_N_CRS,                                          
false,                                             
&l_stiffCsc[l_di].colPtr[0],                        
&l_stiffCsc[l_di].rowIdx[0],                        
&l_stiffCsc[l_di].val[0],                           
TL_N_QTS_E,                                        
TL_N_MDS,                                          
l_maxNzRow+1,                                      
(TL_N_RMS == 0) ? l_maxNzRow+1 : TL_N_MDS,         
0,                                                 
TL_N_MDS,                                          
TL_T_REAL(1.0),                                    
(TL_N_RMS == 0) ? TL_T_REAL(1.0) : TL_T_REAL(0.0), 
LIBXSMM_GEMM_PREFETCH_NONE ); 
}

m_mm.add(  1,                                                 
TL_N_CRS,                                          
true,                                              
&l_starCsrE.rowPtr[0],                              
&l_starCsrE.colIdx[0],                              
&l_starCsrE.val[0],                                 
TL_N_QTS_E,                                        
(TL_N_RMS == 0) ? l_maxNzRow+1 : TL_N_MDS,         
TL_N_QTS_E,                                        
0,                                                 
TL_N_MDS,                                          
(TL_N_RMS == 0) ? l_maxNzRow+1 : TL_N_MDS,         
TL_T_REAL(1.0),                                    
(TL_N_RMS == 0) ? TL_T_REAL(0.0) : TL_T_REAL(1.0), 
LIBXSMM_GEMM_PREFETCH_NONE );

if( TL_N_RMS > 0 ) {
t_matCsr l_starCsrA;
FakeMats< TL_N_DIS >::starCsrA( l_starCsrA );
EDGE_CHECK_EQ( l_starCsrA.val.size(), TL_N_NZS_STAR_A );

t_matCsr l_srcCsrA;
FakeMats< TL_N_DIS >::srcCsrA( l_srcCsrA );
EDGE_CHECK_EQ( l_srcCsrA.val.size(), TL_N_NZS_SRC_A );

m_mm.add( 2,                     
TL_N_CRS,              
true,                  
&l_starCsrA.rowPtr[0], 
&l_starCsrA.colIdx[0], 
&l_starCsrA.val[0],    
TL_N_QTS_M,            
TL_N_MDS,              
TL_N_QTS_E,            
0,                     
TL_N_MDS,              
TL_N_MDS,              
TL_T_REAL(1.0),        
TL_T_REAL(1.0),        
LIBXSMM_GEMM_PREFETCH_NONE );

m_mm.add( 2,                    
TL_N_CRS,             
true,                 
&l_srcCsrA.rowPtr[0], 
&l_srcCsrA.colIdx[0], 
&l_srcCsrA.val[0],    
TL_N_QTS_M,           
TL_N_MDS,             
TL_N_QTS_M,           
0,                    
TL_N_MDS,             
TL_N_MDS,             
TL_T_REAL(1.0),       
TL_T_REAL(1.0),       
LIBXSMM_GEMM_PREFETCH_NONE );
}
}


static void storeStiffSparse( TL_T_REAL     const     i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS],
data::Dynamic       &   io_dynMem,
TL_T_REAL           *   o_stiff[TL_N_DIS]  ) {
unsigned int l_maxNzRow;
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_cscStiff;
getCscStiff( i_stiff,
l_maxNzRow,
l_offsets,
l_nonZeros,
l_cscStiff );

TL_T_REAL * l_stiffRaw = (TL_T_REAL*) io_dynMem.allocate( l_nonZeros.size() * sizeof(TL_T_REAL),
4096,
true );
for( std::size_t l_en = 0; l_en < l_nonZeros.size(); l_en++ ) {
l_stiffRaw[l_en] = l_nonZeros[l_en];
}

EDGE_CHECK_EQ( l_offsets.size(), TL_N_DIS + 1 );

for( unsigned int l_di = 0; l_di < N_DIM; l_di++ ) {
o_stiff[l_di] = l_stiffRaw + l_offsets[l_di];
}
}


public:

VolIntFused( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): VolInt< TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_N_CRS >( i_rfs,
io_dynMem ) {
dg::Basis l_basis( TL_T_EL,
TL_O_SP );

TL_T_REAL l_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS];
l_basis.getStiffMm1Dense( TL_N_MDS,
l_stiff[0][0],
false );

this->storeStiffSparse( l_stiff,
io_dynMem,
m_stiff );

generateKernels( l_stiff );
}


void apply( TL_T_REAL const   i_starE[TL_N_DIS][TL_N_NZS_STAR_E],
TL_T_REAL const (*i_starA)[TL_N_NZS_STAR_A],
TL_T_REAL const (*i_srcA)[TL_N_NZS_SRC_A],
TL_T_REAL const   i_tDofsE[TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL const (*i_tDofsA)[TL_N_QTS_M][TL_N_MDS][TL_N_CRS],
TL_T_REAL         io_dofsE[TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL       (*io_dofsA)[TL_N_QTS_M][TL_N_MDS][TL_N_CRS],
TL_T_REAL         o_scratch[TL_N_QTS_E][TL_N_MDS][TL_N_CRS] ) const {
TL_T_REAL const *l_rfs = this->m_rfs;

TL_T_REAL l_scratch[TL_N_QTS_M][TL_N_MDS][TL_N_CRS];
if( TL_N_RMS > 0 ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ )
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) l_scratch[l_qt][l_md][l_cr] = 0;
}

if( TL_N_RMS > 0 ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) o_scratch[l_qt][l_md][l_cr] = 0;
}

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
if( TL_N_RMS == 0 ) {
m_mm.m_kernels[1][0]( i_starE[l_di],
i_tDofsE[0][0],
o_scratch[0][0] );

m_mm.m_kernels[0][l_di]( o_scratch[0][0],
m_stiff[l_di],
io_dofsE[0][0] );
}
else {
m_mm.m_kernels[0][l_di]( i_tDofsE[0][0],
m_stiff[l_di],
o_scratch[0][0] );

m_mm.m_kernels[1][0]( i_starE[l_di],
o_scratch[0][0],
io_dofsE[0][0] );

m_mm.m_kernels[2][0]( i_starA[l_di],
o_scratch[0][0],
l_scratch[0][0] );
}
}

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_mm.m_kernels[2][1]( i_srcA[l_rm],
i_tDofsA[l_rm][0][0],
io_dofsE[0][0] );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
io_dofsA[l_rm][l_qt][l_md][l_cr] += l_rfs[l_rm] * ( l_scratch[l_qt][l_md][l_cr] - i_tDofsA[l_rm][l_qt][l_md][l_cr] );
}
}
}
}
}
};

#endif