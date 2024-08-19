
#ifndef EDGE_SC_KERNELS_HPP
#define EDGE_SC_KERNELS_HPP

#include "constants.hpp"

namespace edge {
namespace sc {
template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS >
class Kernels;
}
}


template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS >
class edge::sc::Kernels {
private:
static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_VES_FA = C_ENT[TL_T_EL].N_FACE_VERTICES;

static unsigned short const TL_N_SFS = CE_N_SUB_FACES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_SCS = CE_N_SUB_CELLS( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const MM_GR = static_cast< unsigned short >( t_mm::SUB_CELL );

public:

template< typename TL_T_REAL,
typename TL_T_MM >
static void scatter( TL_T_MM   const & i_mm,
TL_T_REAL const   i_dofsDg[TL_N_QTS][TL_N_MDS][TL_N_CRS],
TL_T_REAL const   i_mat[TL_N_MDS][TL_N_SCS],
TL_T_REAL         o_scDofs[TL_N_QTS][TL_N_SCS][TL_N_CRS] ) {
#if defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
i_mm.m_kernels[MM_GR][0]( i_mat[0], i_dofsDg[0][0], o_scDofs[0][0] );
#else
i_mm.m_kernels[MM_GR][0]( i_dofsDg[0][0], i_mat[0], o_scDofs[0][0] );
#endif
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void scatterFa( TL_T_MM   const & i_mm,
TL_T_REAL const   i_dofsDg[TL_N_QTS][TL_N_MDS][TL_N_CRS],
TL_T_REAL const   i_mat[TL_N_MDS][TL_N_SFS],
TL_T_REAL         o_scDofs[TL_N_QTS][TL_N_SFS][TL_N_CRS] ) {
#if defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
i_mm.m_kernels[MM_GR][1]( i_mat[0], i_dofsDg[0][0], o_scDofs[0][0] );
#else
i_mm.m_kernels[MM_GR][1]( i_dofsDg[0][0], i_mat[0], o_scDofs[0][0] );
#endif
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void scatterReplace( TL_T_MM   const & i_mm,
TL_T_REAL const   i_dofsDg[TL_N_QTS][TL_N_MDS][TL_N_CRS],
TL_T_REAL const   i_mat[TL_N_MDS][TL_N_SCS],
TL_T_REAL const   i_dofsSc[TL_N_QTS][TL_N_SCS][TL_N_CRS],
bool      const   i_admP[TL_N_CRS],
TL_T_REAL         o_dofsSc[TL_N_QTS][TL_N_SCS][TL_N_CRS] ) {
scatter( i_mm,
i_dofsDg,
i_mat,
o_dofsSc );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ ) {
#pragma omp simd 
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {  
o_dofsSc[l_qt][l_sc][l_cr] = ( i_admP[l_cr] == false ) ? i_dofsSc[l_qt][l_sc][l_cr] : o_dofsSc[l_qt][l_sc][l_cr];
}
}
}
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void scatterReplaceFa( TL_T_MM   const & i_mm,
TL_T_REAL const   i_dofsDg[TL_N_QTS][TL_N_MDS][TL_N_CRS],
TL_T_REAL const   i_mat[TL_N_MDS][TL_N_SFS],
TL_T_REAL const   i_dofsSc[TL_N_QTS][TL_N_SFS][TL_N_CRS],
bool      const   i_admP[TL_N_CRS],
TL_T_REAL         o_dofsSc[TL_N_QTS][TL_N_SFS][TL_N_CRS] ) {
scatterFa( i_mm,
i_dofsDg,
i_mat,
o_dofsSc );

if( i_admP != nullptr ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_dofsSc[l_qt][l_sf][l_cr] = ( i_admP[l_cr] == false ) ? i_dofsSc[l_qt][l_sf][l_cr] : o_dofsSc[l_qt][l_sf][l_cr];
}
}
}
}
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void gather( TL_T_MM   const &i_mm,
TL_T_REAL const i_scDofs[TL_N_QTS][TL_N_SCS][TL_N_CRS],
TL_T_REAL const i_mat[TL_N_SCS][TL_N_MDS],
TL_T_REAL       o_dgDofs[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
#if defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
i_mm.m_kernels[MM_GR][2]( i_mat[0], i_scDofs[0][0], o_dgDofs[0][0] );
#else
i_mm.m_kernels[MM_GR][2]( i_scDofs[0][0], i_mat[0], o_dgDofs[0][0] );
#endif
}


template< typename TL_T_REAL >
static void gatherSurfDofs( TL_T_REAL      const    i_scDofs[TL_N_QTS][TL_N_SCS][TL_N_CRS],
unsigned short const    i_faSfSc[TL_N_FAS][TL_N_SFS],
unsigned short const    i_scDgAd[TL_N_VES_FA][TL_N_SFS],
unsigned short const    i_vIdElFaEl[TL_N_FAS],
TL_T_REAL            (* o_scSurfDofs [TL_N_FAS])[TL_N_QTS][TL_N_SFS][TL_N_CRS],
bool           const    i_adm[TL_N_CRS] ) {
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
if( o_scSurfDofs[l_fa] != nullptr ) {
unsigned short l_vId = i_vIdElFaEl[l_fa];

for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
unsigned short l_sfRe = i_scDgAd[l_vId][l_sf];

unsigned short l_sc = i_faSfSc[l_fa][l_sfRe];

#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
(*o_scSurfDofs[l_fa])[l_qt][l_sf][l_cr] = ( i_adm[l_cr] == false ) ? i_scDofs[l_qt][l_sc][l_cr] : (*o_scSurfDofs[l_fa])[l_qt][l_sf][l_cr];
}
}
}

}
}
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void sfInt( TL_T_MM   const &i_mm,
TL_T_REAL const i_scFluxes[TL_N_QTS][TL_N_SFS][TL_N_CRS],
TL_T_REAL const i_mat[TL_N_SCS][TL_N_MDS],
TL_T_REAL       o_int[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
#if defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
i_mm.m_kernels[MM_GR][3]( i_mat[0], i_scFluxes[0][0], o_int[0][0] );
#else
i_mm.m_kernels[MM_GR][3]( i_scFluxes[0][0], i_mat[0], o_int[0][0] );
#endif
}


template< typename TL_T_REAL >
static void scExtrema( TL_T_REAL const i_dofsSc[TL_N_QTS][TL_N_SCS][TL_N_CRS],
TL_T_REAL       o_min[TL_N_QTS][TL_N_CRS],
TL_T_REAL       o_max[TL_N_QTS][TL_N_CRS] ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_min[l_qt][l_cr] = std::numeric_limits< TL_T_REAL >::max();
o_max[l_qt][l_cr] = std::numeric_limits< TL_T_REAL >::lowest();
}
}

for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
#pragma omp simd
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_min[l_qt][l_cr] = std::min( o_min[l_qt][l_cr], i_dofsSc[l_qt][l_sc][l_cr] );
o_max[l_qt][l_cr] = std::max( o_max[l_qt][l_cr], i_dofsSc[l_qt][l_sc][l_cr] );
}
}
}
}


template< typename TL_T_REAL,
typename TL_T_MM >
static void dgExtrema( TL_T_MM   const &i_mm,
TL_T_REAL const i_dofsDg[TL_N_QTS][TL_N_MDS][TL_N_CRS],
TL_T_REAL const i_matScatter[TL_N_MDS][TL_N_SCS],
TL_T_REAL       o_subcell[TL_N_QTS][TL_N_SCS][TL_N_CRS],
TL_T_REAL       o_min[TL_N_QTS][TL_N_CRS],
TL_T_REAL       o_max[TL_N_QTS][TL_N_CRS] ) {
scatter( i_mm,
i_dofsDg,
i_matScatter,
o_subcell );

scExtrema( o_subcell,
o_min,
o_max );
}
};

#endif
