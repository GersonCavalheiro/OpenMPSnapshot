
#ifndef EDGE_SEISMIC_KERNELS_TIME_PRED_FUSED_HPP
#define EDGE_SEISMIC_KERNELS_TIME_PRED_FUSED_HPP

#include "TimePred.hpp"
#include "data/MmXsmmFused.hpp"
#include "FakeMats.hpp"

namespace edge {
namespace seismic {
namespace kernels {
template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI,
unsigned short TL_N_CRS >
class TimePredFused;
}
}
}


template< typename       TL_T_REAL,
unsigned short TL_N_RMS,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_O_TI,
unsigned short TL_N_CRS >
class edge::seismic::kernels::TimePredFused: public edge::seismic::kernels::TimePred < TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_O_TI,
TL_N_CRS > {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_E = CE_N_QTS_E( TL_N_DIS );

static unsigned short const TL_N_QTS_M = CE_N_QTS_M( TL_N_DIS );

static unsigned short const TL_N_NZS_STAR_E = CE_N_ENS_STAR_E_SP( TL_N_DIS );

static unsigned short const TL_N_NZS_STAR_A = CE_N_ENS_STAR_A_SP( TL_N_DIS );

static unsigned short const TL_N_NZS_SRC_A = CE_N_ENS_SRC_A_SP( TL_N_DIS );

TL_T_REAL *m_stiffT[CE_MAX(TL_O_TI-1,1)][TL_N_DIS] = {};

edge::data::MmXsmmFused< TL_T_REAL > m_mm;


static void inline zero( TL_T_REAL o_mat[TL_N_QTS_E][TL_N_MDS][TL_N_CRS] ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
for( unsigned short l_cfr = 0; l_cfr < TL_N_CRS; l_cfr++ ) {
o_mat[l_qt][l_md][l_cfr] = 0;
}
}
}
}


static void getCscStiffT( TL_T_REAL               const   i_stiffT[TL_N_DIS][TL_N_MDS][TL_N_MDS],
std::vector< size_t >         & o_maxNzCols,
std::vector< size_t >         & o_offsets,
std::vector< TL_T_REAL >      & o_nonZeros,
std::vector< t_matCsc >       & o_mats ) {
std::string l_cscFillIn = "none";

if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_KNM ) {
l_cscFillIn = "qfma";
}

o_maxNzCols.resize( 0 );
o_offsets.resize( 0 );
o_offsets.push_back( 0 );
o_nonZeros.resize( 0 );
o_mats.resize( 0 );

t_matCrd l_stiffTCrd[TL_N_DIS];
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
edge::linalg::Matrix::denseToCrd< TL_T_REAL >( TL_N_MDS,
TL_N_MDS,
i_stiffT[l_di][0],
l_stiffTCrd[l_di],
TOL.BASIS );
}

unsigned int l_nzBl[2][2][2];
l_nzBl[0][0][0] = l_nzBl[0][1][0] = 0;
l_nzBl[0][0][1] = l_nzBl[0][1][1] = TL_N_MDS-1;

for( unsigned short l_de = 1; l_de < TL_O_TI; l_de++ ) {
unsigned int l_maxNzCol = 0;

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
edge::linalg::Matrix::getBlockNz( l_stiffTCrd[l_di], l_nzBl[0], l_nzBl[1] );
l_maxNzCol = std::max( l_maxNzCol, l_nzBl[1][1][1] );
}
o_maxNzCols.push_back( l_maxNzCol );

t_matCsc l_stiffTCsc[TL_N_DIS];
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
edge::linalg::Matrix::denseToCsc< TL_T_REAL >( TL_N_MDS,
TL_N_MDS,
i_stiffT[l_di][0],
l_stiffTCsc[l_di],
TOL.BASIS,
l_nzBl[0][0][1]+1,
l_maxNzCol+1,
l_cscFillIn );
o_mats.push_back( l_stiffTCsc[l_di] );
}

if( l_de == 1 || l_maxNzCol < l_nzBl[0][0][1] ) {
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
for( unsigned int l_nz = 0; l_nz < l_stiffTCsc[l_di].val.size(); l_nz++ ) {
o_nonZeros.push_back( l_stiffTCsc[l_di].val[l_nz] );
}
o_offsets.push_back( o_offsets.back() + l_stiffTCsc[l_di].val.size() );
}
}
else {
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
o_offsets.push_back( o_offsets.back() );
}
}

#ifdef PP_T_BASIS_HIERARCHICAL
EDGE_CHECK_EQ( l_maxNzCol+1, CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP-l_de ) );
#endif

l_nzBl[0][0][1] = l_maxNzCol;
}
}


void generateKernels( TL_T_REAL const i_stiffT[TL_N_DIS][TL_N_MDS][TL_N_MDS] ) {
std::vector< size_t > l_maxNzCols;
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_cscStiffT;
getCscStiffT( i_stiffT,
l_maxNzCols,
l_offsets,
l_nonZeros,
l_cscStiffT );

t_matCsr l_starCsrE;
FakeMats< TL_N_DIS >::starCsrE( l_starCsrE );
EDGE_CHECK_EQ( l_starCsrE.val.size(), TL_N_NZS_STAR_E );

for( unsigned short l_de = 1; l_de < TL_O_TI; l_de++ ) {
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
m_mm.add( 0,                                                
TL_N_CRS,                                         
false,                                            
&l_cscStiffT[(l_de-1)*TL_N_DIS + l_di].colPtr[0], 
&l_cscStiffT[(l_de-1)*TL_N_DIS + l_di].rowIdx[0], 
&l_cscStiffT[(l_de-1)*TL_N_DIS + l_di].val[0],    
TL_N_QTS_E,                                       
l_maxNzCols[l_de-1]+1,                            
(l_de == 1) ? TL_N_MDS : l_maxNzCols[l_de-2]+1,   
TL_N_MDS,                                         
0,                                                
l_maxNzCols[l_de-1]+1,                            
TL_T_REAL(1.0),                                   
TL_T_REAL(0.0),                                   
LIBXSMM_GEMM_PREFETCH_NONE );
}
m_mm.add( 1,                     
TL_N_CRS,              
true,                  
&l_starCsrE.rowPtr[0], 
&l_starCsrE.colIdx[0], 
&l_starCsrE.val[0],    
TL_N_QTS_E,            
l_maxNzCols[l_de-1]+1, 
TL_N_QTS_E,            
0,                     
l_maxNzCols[l_de-1]+1, 
TL_N_MDS,              
TL_T_REAL(1.0),        
TL_T_REAL(1.0),        
LIBXSMM_GEMM_PREFETCH_NONE );
}

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
l_maxNzCols[0]+1,      
TL_N_QTS_E,            
0,                     
l_maxNzCols[0]+1,      
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


static void storeStiffTSparse( TL_T_REAL     const     i_stiffT[TL_N_DIS][TL_N_MDS][TL_N_MDS],
data::Dynamic       &   io_dynMem,
TL_T_REAL           *   o_stiffT[CE_MAX(TL_O_TI-1,1)][TL_N_DIS]  ) {
std::vector< size_t > l_maxNzCols;
std::vector< size_t > l_offsets;
std::vector< TL_T_REAL > l_nonZeros;
std::vector< t_matCsc > l_cscStiffT;
getCscStiffT( i_stiffT,
l_maxNzCols,
l_offsets,
l_nonZeros,
l_cscStiffT );

TL_T_REAL * l_stiffTRaw = (TL_T_REAL*) io_dynMem.allocate( l_nonZeros.size() * sizeof(TL_T_REAL),
4096,
true );
for( std::size_t l_en = 0; l_en < l_nonZeros.size(); l_en++ ) {
l_stiffTRaw[l_en] = -l_nonZeros[l_en];
}

EDGE_CHECK_EQ( l_offsets.size(), (TL_O_TI-1)*TL_N_DIS+1 );

unsigned short l_mat = 0;
for( unsigned int l_de = 1; l_de < ORDER; l_de++ ) {
for( unsigned int l_di = 0; l_di < N_DIM; l_di++ ) {
o_stiffT[(l_de-1)][l_di] = l_stiffTRaw + l_offsets[l_mat];
l_mat++;
}
}
}

public:

TimePredFused( TL_T_REAL     const * i_rfs,
data::Dynamic       & io_dynMem ): TimePred < TL_T_REAL,
TL_N_RMS,
TL_T_EL,
TL_O_SP,
TL_O_TI,
TL_N_CRS >( i_rfs,
io_dynMem ) {
dg::Basis l_basis( TL_T_EL,
TL_O_SP );

TL_T_REAL l_stiffT[TL_N_DIS][TL_N_MDS][TL_N_MDS];
l_basis.getStiffMm1Dense( TL_N_MDS,
l_stiffT[0][0],
true );


this->storeStiffTSparse( l_stiffT,
io_dynMem,
m_stiffT );

generateKernels( l_stiffT );
};


void ck( TL_T_REAL         i_dT,
TL_T_REAL const   i_starE[TL_N_DIS][TL_N_NZS_STAR_E],
TL_T_REAL const (*i_starA)[TL_N_NZS_STAR_A],
TL_T_REAL const (*i_srcA)[TL_N_NZS_SRC_A],
TL_T_REAL const   i_dofsE[TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL const (*i_dofsA)[TL_N_QTS_M][TL_N_MDS][TL_N_CRS],
TL_T_REAL         o_scratch[TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL         o_derE[TL_O_TI][TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL       (*o_derA)[TL_O_TI][TL_N_QTS_M][TL_N_MDS][TL_N_CRS],
TL_T_REAL         o_tIntE[TL_N_QTS_E][TL_N_MDS][TL_N_CRS],
TL_T_REAL       (*o_tIntA)[TL_N_QTS_M][TL_N_MDS][TL_N_CRS] ) const {
TL_T_REAL const *l_rfs = this->m_rfs;

TL_T_REAL l_scalar = i_dT;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
for( unsigned short l_cfr = 0; l_cfr < TL_N_CRS; l_cfr++ ) {
o_derE[0][l_qt][l_md][l_cfr] = i_dofsE[l_qt][l_md][l_cfr];
o_tIntE[l_qt][l_md][l_cfr]   = l_scalar * i_dofsE[l_qt][l_md][l_cfr];
}
}
}

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_derA[l_rm][0][l_qt][l_md][l_cr] = i_dofsA[l_rm][l_qt][l_md][l_cr];
o_tIntA[l_rm][l_qt][l_md][l_cr] = l_scalar * i_dofsA[l_rm][l_qt][l_md][l_cr];
}
}
}
}

for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
unsigned short l_re = (TL_N_RMS == 0) ? l_de : 1;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ )
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) o_derE[l_de][l_qt][l_md][l_cr] = 0;

TL_T_REAL l_scratch[TL_N_QTS_M][TL_N_MDS][TL_N_CRS];
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ )
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) l_scratch[l_qt][l_md][l_cr] = 0;

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
if( TL_T_EL == TET4 || TL_T_EL == TRIA3 ) {}
else {
zero( o_scratch );
}

m_mm.m_kernels[0][(l_re-1)*(TL_N_DIS)+l_di]( o_derE[l_de-1][0][0],
m_stiffT[l_re-1][l_di],
o_scratch[0][0] );
m_mm.m_kernels[1][l_re-1]( i_starE[l_di],
o_scratch[0][0],
o_derE[l_de][0][0] );

if( TL_N_RMS > 0 ) {
m_mm.m_kernels[2][0]( i_starA[l_di],
o_scratch[0][0],
l_scratch[0][0] );
}
}

l_scalar *= i_dT / (l_de+1);

for( unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++ ) {
m_mm.m_kernels[2][1]( i_srcA[l_rm],
o_derA[l_rm][l_de-1][0][0],
o_derE[l_de][0][0] );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_derA[l_rm][l_de][l_qt][l_md][l_cr] = l_rfs[l_rm] * ( l_scratch[l_qt][l_md][l_cr] + o_derA[l_rm][l_de-1][l_qt][l_md][l_cr] );
o_tIntA[l_rm][l_qt][l_md][l_cr] += l_scalar * o_derA[l_rm][l_de][l_qt][l_md][l_cr];
}
}
}
}

unsigned short l_nCpMds = (TL_N_RMS == 0) ? CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ) : TL_N_MDS;

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_E; l_qt++ ) {
for( unsigned short l_md = 0; l_md < l_nCpMds; l_md++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_tIntE[l_qt][l_md][l_cr] += l_scalar * o_derE[l_de][l_qt][l_md][l_cr];
}
}
}
}
}
};

#endif