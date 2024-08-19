
#ifndef EDGE_SC_INIT_HPP
#define EDGE_SC_INIT_HPP

#include "constants.hpp"
#include "io/logging.h"


namespace edge {
namespace pre {
namespace sc {
extern const unsigned short *g_scsvRaw;
extern const std::size_t     g_scsvSize;

extern const unsigned short *g_scsfscRaw;
extern const std::size_t     g_scsfscSize;

extern const unsigned short *g_sctysfRaw;
extern const std::size_t     g_sctysfSize;

extern const unsigned short *g_scdgadRaw;
extern const std::size_t     g_scdgadSize;

extern const double      *g_svcrdsRaw;
extern const std::size_t  g_svcrdsSize;

extern const double      *g_scatterRaw;
extern const std::size_t  g_scatterSize;

extern const double      *g_scattersurfRaw;
extern const std::size_t  g_scattersurfSize;

extern const double      *g_gatherRaw;
extern const std::size_t  g_gatherSize;

extern const double      *g_sfintRaw;
extern const std::size_t  g_sfintSize;
}
}
}

namespace edge {
namespace sc {
template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS  >
class Init;
}
}


template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS  >
class edge::sc::Init {
private:
static unsigned short const TL_N_DIS = C_ENT[ TL_T_EL ].N_DIM;

static unsigned short const TL_N_VES_FA = C_ENT[ TL_T_EL ].N_FACE_VERTICES;

static unsigned short const TL_N_VES_EL = C_ENT[ TL_T_EL ].N_VERTICES;

static unsigned short const TL_N_FAS = C_ENT[ TL_T_EL ].N_FACES;

static unsigned short const TL_N_SVS = CE_N_SUB_VERTICES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_SFS = CE_N_SUB_FACES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_SCS = CE_N_SUB_CELLS( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_SCASU = (TL_N_DIS < 3) ? 2*TL_N_FAS : TL_N_FAS+TL_N_VES_FA*TL_N_FAS;


template< typename TL_T_LID >
static void getFaSfSc( TL_T_LID const i_scSfSc[ TL_N_SCS + TL_N_FAS * TL_N_SFS ][ TL_N_FAS ],
TL_T_LID       o_faSfSc[TL_N_FAS][TL_N_SFS] ) {
static_assert(
std::is_same< decltype( pre::sc::g_scsfscRaw ),
const unsigned short* >::value,
"g_scsfscRaw is assumed to be const unsigned short*" );

for( unsigned short l_f1 = 0; l_f1 < TL_N_FAS; l_f1++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
TL_T_LID l_sc = i_scSfSc[ TL_N_SCS + l_f1 * TL_N_SFS + l_sf ][0];
o_faSfSc[l_f1][l_sf] = l_sc;
EDGE_CHECK_NE( o_faSfSc[l_f1][l_sf],
std::numeric_limits< unsigned short >::max() );

for( unsigned short l_f2 = 1; l_f2 < TL_N_FAS; l_f2++ ) {
EDGE_CHECK_EQ( i_scSfSc[ TL_N_SCS + l_f1 * TL_N_SFS + l_sf ][l_f2],
std::numeric_limits< unsigned short >::max() );
}
}
}
}

public:

template< typename TL_T_LID >
static void connect( t_connect< TL_T_LID,
TL_T_EL,
TL_O_SP  > &o_conn ) {
std::size_t l_size;

unsigned short l_nScs = TL_N_SCS + TL_N_FAS * TL_N_SFS;

unsigned short const * l_ptr;


l_size  = l_nScs;
l_size *= TL_N_VES_EL;
EDGE_CHECK_EQ( edge::pre::sc::g_scsvSize,
l_size );

l_ptr = edge::pre::sc::g_scsvRaw;

for( unsigned short l_sc = 0; l_sc < l_nScs; l_sc++ ) {
for( unsigned short l_ve = 0; l_ve < TL_N_VES_EL; l_ve++ ) {
o_conn.scSv[l_sc][l_ve] = *l_ptr;
l_ptr++;
}
}


l_size = l_nScs;
l_size *= TL_N_FAS;
EDGE_CHECK_EQ( edge::pre::sc::g_scsfscSize,
l_size );

l_ptr = edge::pre::sc::g_scsfscRaw;

for( unsigned short l_sc = 0; l_sc < l_nScs; l_sc++ ) {
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
o_conn.scSfSc[l_sc][l_fa] = *l_ptr;
l_ptr++;
}
}


getFaSfSc( o_conn.scSfSc,
o_conn.faSfSc );


l_size =  TL_N_SCS;
l_size *= TL_N_FAS;
EDGE_CHECK_EQ( edge::pre::sc::g_sctysfSize,
l_size );

l_ptr = edge::pre::sc::g_sctysfRaw;

for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ ) {
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
o_conn.scTySf[l_sc][l_fa] = *l_ptr;
l_ptr++;
}
}


const unsigned short l_or = (TL_N_DIS == 3) ? TL_N_VES_FA : 1;

l_size = l_or;
l_size *= TL_N_SFS;
EDGE_CHECK_EQ( edge::pre::sc::g_scdgadSize,
l_size );
l_ptr = edge::pre::sc::g_scdgadRaw;

for( unsigned short l_ve = 0; l_ve < l_or; l_ve++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
o_conn.scDgAd[l_ve][l_sf] = *l_ptr;
l_ptr++;
}
}
}


template < typename TL_T_REAL >
static void svChars( t_svChars< TL_T_REAL,
TL_T_EL > o_svChars[TL_N_SVS] ) {
std::size_t l_size;

double const * l_ptr;


l_size = TL_N_SVS * TL_N_DIS;
EDGE_CHECK_EQ( edge::pre::sc::g_svcrdsSize,
l_size );

l_ptr = edge::pre::sc::g_svcrdsRaw;

for( unsigned short l_sv = 0; l_sv < TL_N_SVS; l_sv++ ) {
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
o_svChars[l_sv].coords[l_di] = *l_ptr;
l_ptr++;
}
}
}


template < typename TL_T_REAL >
static void ops( t_ops< TL_T_REAL,
TL_T_EL,
TL_O_SP > &o_ops ) {
std::size_t l_size;

double const * l_ptr;


l_size = TL_N_MDS;
l_size *= TL_N_SCS;
EDGE_CHECK_EQ( edge::pre::sc::g_scatterSize,
l_size );

l_ptr = edge::pre::sc::g_scatterRaw;

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ ) {
o_ops.scatter[l_md][l_sc] = *l_ptr;
l_ptr++;
}
}



l_size  = TL_N_MDS;
l_size *= TL_N_SCASU;
l_size *= TL_N_SFS;
EDGE_CHECK_EQ( edge::pre::sc::g_scattersurfSize,
l_size );

l_ptr = edge::pre::sc::g_scattersurfRaw;

for( unsigned short l_fa = 0; l_fa < TL_N_SCASU; l_fa++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
o_ops.scatterSurf[l_fa][l_md][l_sf] = *l_ptr;
l_ptr++;
}
}
}


l_size = TL_N_SCS;
l_size *= TL_N_MDS;
EDGE_CHECK_EQ( edge::pre::sc::g_gatherSize,
l_size );

l_ptr = edge::pre::sc::g_gatherRaw;

for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_ops.gather[l_sc][l_md] = *l_ptr;
l_ptr++;
}
}


l_size = TL_N_FAS * TL_N_SFS;
l_size *= TL_N_MDS;
EDGE_CHECK_EQ( edge::pre::sc::g_sfintSize,
l_size );

l_ptr = edge::pre::sc::g_sfintRaw;

for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_ops.sfInt[l_fa][l_sf][l_md] = *l_ptr;
l_ptr++;
}
}
}
}


template< typename TL_T_LID,
typename TL_T_REAL >
static void data( TL_T_LID        i_nLim,
TL_T_LID        i_nLimP,
TL_T_LID        i_nExt,
TL_T_REAL    (* o_dofs)[TL_N_QTS][TL_N_SCS][TL_N_CRS],
TL_T_REAL    (* o_tDofsRaw[2])[TL_N_QTS][TL_N_SFS][TL_N_CRS],
TL_T_REAL    (* o_ext[2])[2][TL_N_QTS][TL_N_CRS],
bool         (* o_adm[3])[TL_N_CRS],
bool         (* o_lock)[TL_N_CRS],
unsigned int (* o_limSync)[TL_N_CRS] ) {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_li = 0; l_li < i_nLim; l_li++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
for( unsigned short l_sc = 0; l_sc < TL_N_SCS; l_sc++ )
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_dofs[l_li][l_qt][l_sc][l_cr] = 0;
for( unsigned short l_ad = 0; l_ad < 4; l_ad++ )
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_adm[l_ad][l_li][l_cr] = true;
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_lock[l_li][l_cr] = false;
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_limSync[l_li][l_cr] = 0;
}

for( unsigned short l_bu = 0; l_bu < 2; l_bu++ )
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_lp = 0; l_lp < i_nLimP; l_lp++ )
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ )
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ )
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_tDofsRaw[l_bu][l_lp*TL_N_FAS + l_fa][l_qt][l_sf][l_cr] = 0;

for( unsigned short l_e1 = 0; l_e1 < 2; l_e1++ )
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_ex = 0; l_ex < i_nExt; l_ex++ )
for( unsigned short l_e2 = 0; l_e2 < 2; l_e2++ )
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_ext[l_e1][l_ex][l_e2][l_qt][l_cr] = 0;
}


template< typename TL_T_LID >
static void liDoLiDu( t_enLayout const  & i_layoutLim,
TL_T_LID   const  * i_liLp,
TL_T_LID          * i_lpEl,
TL_T_LID   const  * i_elDaMe,
TL_T_LID   const  * i_elMeDa,
TL_T_LID         (* o_liDoLiDu)[TL_N_FAS] ) {
for( TL_T_LID l_li = 0; l_li < i_layoutLim.nEnts; l_li++ )
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ )
o_liDoLiDu[l_li][l_fa] = std::numeric_limits< TL_T_LID >::max();

TL_T_LID l_seFirst = i_layoutLim.timeGroups[0].inner.first + i_layoutLim.timeGroups[0].inner.size;
TL_T_LID l_seSize  = i_layoutLim.timeGroups[0].nEntsOwn - i_layoutLim.timeGroups[0].inner.size;

for( TL_T_LID l_l1 = l_seFirst; l_l1 < l_seFirst+l_seSize; l_l1++ ) {
TL_T_LID l_lp = i_liLp[l_l1];
TL_T_LID l_el = i_lpEl[l_lp];

TL_T_LID l_elDaMe = i_elDaMe[l_el];
TL_T_LID l_elDo   = i_elMeDa[l_elDaMe];

if( l_elDo != l_el ) {
for( TL_T_LID l_l2 = l_seFirst; l_l2 < l_seFirst+l_seSize; l_l2++ ) {
TL_T_LID l_lpAd = i_liLp[l_l2];
TL_T_LID l_elAd = i_lpEl[l_lpAd];

if( l_elAd == l_elDo ) {
for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
if( o_liDoLiDu[l_l2][l_fa] == std::numeric_limits< TL_T_LID >::max() ) {
o_liDoLiDu[l_l2][l_fa] = l_l1;
break;
}
EDGE_CHECK_NE( l_fa, TL_N_FAS-1 );
}
}

}
}
}
}
};

#endif
