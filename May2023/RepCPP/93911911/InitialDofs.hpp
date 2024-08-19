
#ifndef EDGE_INITIAL_DOFS_HPP
#define EDGE_INITIAL_DOFS_HPP

#include "constants.hpp"
#include "io/logging.h"
#include "data/Expression.hpp"
#include "mesh/common.hpp"
#include "dg/QuadraturePoints.h"
#include "sc/SubGrid.hpp"

namespace edge {
namespace setups {
template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS >
class InitialDofs;
}
}


template< t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_QTS,
unsigned short TL_N_CRS >
class edge::setups::InitialDofs {
private:
static unsigned short const TL_N_DIMS = C_ENT[ TL_T_EL ].N_DIM;

static unsigned short const TL_N_VES = C_ENT[ TL_T_EL ].N_VERTICES;

static unsigned short const TL_N_SCS = CE_N_SUB_CELLS( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned int const TL_N_QPS1 = CE_N_QUAD_POINTS( TL_T_EL, TL_O_SP+1 );


template< typename TL_T_LID,
typename TL_T_REAL,
typename TL_T_VE_CHARS >
static void qps( unsigned short                   i_order,
TL_T_LID                         i_el,
TL_T_LID                 const (*i_elVe)[TL_N_VES],
TL_T_VE_CHARS            const  *i_veChars,
std::vector< TL_T_REAL >         o_pts[3],
std::vector< TL_T_REAL >        &o_wes ) {
TL_T_REAL l_veCrds[3][TL_N_VES];

for( unsigned short l_di = 0; l_di < 3; l_di++ )
for( unsigned short l_ve = 0; l_ve < TL_N_VES; l_ve++ ) l_veCrds[l_di][l_ve] = 0;

mesh::common<
TL_T_EL
>::getElVeCrds( i_el,
i_elVe,
i_veChars,
l_veCrds );

dg::QuadraturePoints::getQpts( TL_T_EL,
i_order,
l_veCrds,
o_pts[0], o_pts[1], o_pts[2], o_wes );
}


template< typename TL_T_REAL >
static void bc( std::string                         const i_exprStrs[TL_N_CRS],
TL_T_REAL                                 io_crds[TL_N_DIMS],
TL_T_REAL                                 io_qts[TL_N_CRS][TL_N_QTS],
edge::data::Expression< TL_T_REAL >       io_exprs[TL_N_CRS] ) {
for( unsigned short l_di = 0; l_di < TL_N_DIMS; l_di++ )
io_crds[l_di] = std::numeric_limits< TL_T_REAL >::max();

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
io_qts[l_cr][l_qt] = std::numeric_limits< TL_T_REAL >::max();

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
io_exprs[l_cr].bindCrds( io_crds, TL_N_DIMS );
io_exprs[l_cr].bind( "q", io_qts[l_cr], TL_N_QTS );

io_exprs[l_cr].compile( i_exprStrs[l_cr] );
}
}

public:

template< typename TL_T_LID,
typename TL_T_REAL,
typename TL_T_VE_CHARS >
static void dg( TL_T_LID              i_first,
TL_T_LID              i_size,
std::string   const   i_exprStrs[TL_N_CRS],
dg::Basis     const  &i_basis,
TL_T_LID            (*i_elVe)[TL_N_VES],
TL_T_VE_CHARS const  *i_veChars,
TL_T_REAL           (*o_dofs)[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
#ifdef PP_USE_OMP
#pragma omp parallel
#endif
{
TL_T_REAL l_crds[TL_N_DIMS];

TL_T_REAL l_qts[TL_N_CRS][TL_N_QTS];

edge::data::Expression< TL_T_REAL > l_exprs[TL_N_CRS];

bc( i_exprStrs, l_crds, l_qts, l_exprs );

#ifdef PP_USE_OMP
#pragma omp for
#endif
for( TL_T_LID l_el = i_first; l_el < i_first+i_size; l_el++ ) {
std::vector< TL_T_REAL > l_pts[3], l_wes;
qps( TL_O_SP+1,
l_el,
i_elVe,
i_veChars,
l_pts,
l_wes );
EDGE_CHECK( l_wes.size() == TL_N_QPS1 ); 

TL_T_REAL l_q0[TL_N_CRS][TL_N_QTS][TL_N_QPS1];

for( unsigned short l_qp = 0; l_qp < TL_N_QPS1; l_qp++ ) {
for( unsigned short l_di = 0; l_di < TL_N_DIMS; l_di++ )
l_crds[l_di] = l_pts[l_di][l_qp];

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
l_qts[l_cr][l_qt]  = 0;

l_exprs[l_cr].eval();

for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
l_q0[l_cr][l_qt][l_qp] = l_qts[l_cr][l_qt];
}
}

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
TL_T_REAL l_mds[TL_N_MDS];

i_basis.qpts2modal( l_q0[l_cr][l_qt],
TL_O_SP+1,
l_mds );

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_dofs[l_el][l_qt][l_md][l_cr] = l_mds[l_md];
}

}
}

}
}
}


template< typename TL_T_LID,
typename TL_T_REAL,
typename TL_T_VE_CHARS,
typename TL_T_EL_CHARS >
static void err( TL_T_LID               i_first,
TL_T_LID               i_size,
std::string    const   i_exprStrs[TL_N_CRS],
dg::Basis      const  &i_basis,
TL_T_LID       const (*i_elVe)[TL_N_VES],
TL_T_VE_CHARS  const  *i_veChars,
TL_T_EL_CHARS  const  *i_elChars,
TL_T_REAL      const (*i_dofsDg)[TL_N_QTS][TL_N_MDS][TL_N_CRS],
double                 o_l1[TL_N_QTS][TL_N_CRS],
double                 o_l2p2[TL_N_QTS][TL_N_CRS],
double                 o_lInf[TL_N_QTS][TL_N_CRS] ) {
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
o_l1[l_qt][l_cr] = o_l2p2[l_qt][l_cr] = o_lInf[l_qt][l_cr] = 0;

double l_cE[TL_N_DIMS];
double l_qE[TL_N_CRS][TL_N_QTS];

edge::data::Expression< double > l_exprs[TL_N_CRS];

bc( i_exprStrs, l_cE, l_qE, l_exprs );

std::vector< double > l_ptsR[3], l_wesR;
dg::QuadraturePoints::getQpts( TL_T_EL,
TL_O_SP+1,
C_REF_ELEMENT.VE.ENT[TL_T_EL],
l_ptsR[0], l_ptsR[1], l_ptsR[2], l_wesR );
EDGE_CHECK( l_wesR.size() == TL_N_QPS1 ); 

for( TL_T_LID l_el = i_first; l_el < i_first+i_size; l_el++ ) {
std::vector< double > l_ptsP[3], l_wesP;
qps( TL_O_SP+1,
l_el,
i_elVe,
i_veChars,
l_ptsP,
l_wesP );
EDGE_CHECK_EQ( l_wesP.size(), TL_N_QPS1 );

double l_qR[TL_N_CRS][TL_N_QTS][TL_N_QPS1];
for( unsigned short l_qp = 0; l_qp < TL_N_QPS1; l_qp++ ) {
for( unsigned short l_di = 0; l_di < TL_N_DIMS; l_di++ )
l_cE[l_di] = l_ptsP[l_di][l_qp];

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
l_exprs[l_cr].eval();
for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
l_qR[l_cr][l_qt][l_qp] = l_qE[l_cr][l_qt];
}
}

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
TL_T_REAL l_qN[TL_N_QTS][TL_N_QPS1];

for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
TL_T_REAL l_mds[TL_N_MDS];
for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
l_mds[l_md] = i_dofsDg[l_el][l_qt][l_md][l_cr];

for( unsigned short l_qp = 0; l_qp < TL_N_QPS1; l_qp++ )
l_qN[l_qt][l_qp] = i_basis.modal2refPtVal( TL_O_SP+1, l_qp, l_mds );
}

for( unsigned short l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
for( unsigned short l_qp = 0; l_qp < TL_N_QPS1; l_qp++ ) {
double l_diff = std::abs( l_qN[l_qt][l_qp] - l_qR[l_cr][l_qt][l_qp] );
o_l1[l_qt][l_cr]   += l_diff *          l_wesP[l_qp];
o_l2p2[l_qt][l_cr] += l_diff * l_diff * l_wesP[l_qp];
o_lInf[l_qt][l_cr]  = std::max( o_lInf[l_qt][l_cr], l_diff );
}
}
}
}
}
};

#endif
