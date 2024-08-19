
#ifndef EDGE_SEISMIC_SOLVERS_ADER_DG_INIT_HPP
#define EDGE_SEISMIC_SOLVERS_ADER_DG_INIT_HPP

#include "../setups/Elasticity.h"
#include "../setups/ViscoElasticity.h"
#include "mesh/common.hpp"

namespace edge {
namespace seismic {
namespace solvers {
template< t_entityType TL_T_EL,
bool         TL_MATS_SP >
class AderDgInit;
}
}
}


template< t_entityType TL_T_EL,
bool         TL_MATS_SP >
class edge::seismic::solvers::AderDgInit {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_VES_EL = C_ENT[TL_T_EL].N_VERTICES;

static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_ENS_STAR_E = (TL_MATS_SP) ? CE_N_ENS_STAR_E_SP( TL_N_DIS )
: CE_N_ENS_STAR_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_E = CE_N_ENS_FS_E_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_SRC_A = (TL_MATS_SP) ? CE_N_ENS_SRC_A_SP( TL_N_DIS )
: CE_N_ENS_SRC_A_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_STAR_A = (TL_MATS_SP) ? CE_N_ENS_STAR_A_SP( TL_N_DIS )
: CE_N_ENS_STAR_A_DE( TL_N_DIS );

static unsigned short const TL_N_ENS_FS_A = CE_N_ENS_FS_A_DE( TL_N_DIS );

public:

template< typename TL_T_LID,
typename TL_T_REAL >
static void initSrcA( TL_T_LID          i_nEls,
unsigned short    i_nRms,
double            i_freqCen,
double            i_freqRat,
t_bgPars        * io_bgPars,
TL_T_REAL      (* o_srcA)[TL_N_ENS_SRC_A] ) {
double l_lameE[2] = { std::numeric_limits< double >::max(), std::numeric_limits< double >::max() };

for( TL_T_LID l_el = 0; l_el < i_nEls; l_el++ ) {
seismic::setups::ViscoElasticity::src( i_nRms,
i_freqCen,
i_freqRat,
io_bgPars[l_el].qp,
io_bgPars[l_el].qs,
io_bgPars[l_el].lam,
io_bgPars[l_el].mu,
l_lameE[0],
l_lameE[1],
(o_srcA+l_el*std::size_t(i_nRms)) );

io_bgPars[l_el].lam = l_lameE[0];
io_bgPars[l_el].mu  = l_lameE[1];
}
}


template< typename TL_T_LID,
typename TL_T_REAL >
static void initStar( TL_T_LID               i_nEls,
TL_T_LID      const (* i_elVe)[TL_N_VES_EL],
t_vertexChars const  * i_veChars,
t_bgPars      const  * i_bgPars,
TL_T_REAL           (* o_starE)[TL_N_DIS][TL_N_ENS_STAR_E],
TL_T_REAL           (* o_starA)[TL_N_DIS][TL_N_ENS_STAR_A] ) {
for( TL_T_LID l_el = 0; l_el < i_nEls; l_el++ ) {
double l_veCrds[TL_N_DIS][TL_N_VES_EL];
mesh::common< TL_T_EL >::getElVeCrds( l_el,
i_elVe,
i_veChars,
l_veCrds );

double l_jac[TL_N_DIS][TL_N_DIS];
double l_jacInv[TL_N_DIS][TL_N_DIS];
linalg::Mappings::evalJac( TL_T_EL, l_veCrds[0], l_jac[0] );

linalg::Matrix::inv( l_jac, l_jacInv );

double l_starE[TL_N_DIS][TL_N_ENS_STAR_E];
seismic::setups::Elasticity::star( i_bgPars[l_el].rho,
i_bgPars[l_el].lam,
i_bgPars[l_el].mu,
l_jacInv,
l_starE );
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ )
for( unsigned short l_en = 0; l_en < TL_N_ENS_STAR_E; l_en++ )
o_starE[l_el][l_di][l_en] = l_starE[l_di][l_en];

if( o_starA != nullptr ) {
double l_starA[TL_N_DIS][TL_N_ENS_STAR_A];
seismic::setups::ViscoElasticity::star( l_jacInv,
l_starA );
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ )
for( unsigned short l_en = 0; l_en < TL_N_ENS_STAR_A; l_en++ )
o_starA[l_el][l_di][l_en] = l_starA[l_di][l_en];
}
}
}


template< typename TL_T_REAL >
static void setUpFs( double    i_rhoL,
double    i_rhoR,
double    i_lamL,
double    i_lamR,
double    i_muL,
double    i_muR,
double    i_nx,
double    i_ny,
TL_T_REAL o_fsEl[5*5],
TL_T_REAL o_fsEr[5*5],
TL_T_REAL o_fsAl[3*5],
TL_T_REAL o_fsAr[3*5],
bool      i_freeSurface ) {
double l_tE[5][5];
double l_tA[3][3];
double l_tm1[5][5];

seismic::common::setupTrafo2d( i_nx, i_ny, l_tE );

for( unsigned short l_q0 = 0; l_q0 < 3; l_q0++ )
for( unsigned short l_q1 = 0; l_q1 < 3; l_q1++ )
l_tA[l_q0][l_q1] = l_tE[l_q0][l_q1];

seismic::common::setupTrafoInv2d( i_nx, i_ny, l_tm1 );

double l_fsE[2][5*5];
setups::Elasticity::fs( i_rhoL,
i_rhoR,
i_lamL,
i_lamR,
i_muL,
i_muR,
l_tE,
l_tm1,
l_fsE[0],
l_fsE[1],
i_freeSurface );
for( unsigned short l_en = 0; l_en < 5*5; l_en++ ) {
o_fsEl[l_en] = l_fsE[0][l_en];
o_fsEr[l_en] = l_fsE[1][l_en];
}

if( o_fsAl != nullptr && o_fsAr != nullptr ) {
double l_fsA[2][3*5];
setups::ViscoElasticity::fs( i_rhoL,
i_rhoR,
i_lamL,
i_lamR,
i_muL,
i_muR,
l_tA,
l_tm1,
l_fsA[0],
l_fsA[1],
i_freeSurface );

for( unsigned short l_en = 0; l_en < 3*5; l_en++ ) {
o_fsAl[l_en] = l_fsA[0][l_en];
o_fsAr[l_en] = l_fsA[1][l_en];
}
}
}


template< typename TL_T_REAL >
static void setUpFs( double    i_rhoL,
double    i_rhoR,
double    i_lamL,
double    i_lamR,
double    i_muL,
double    i_muR,
double    i_nx,
double    i_ny,
double    i_nz,
double    i_sx,
double    i_sy,
double    i_sz,
double    i_tx,
double    i_ty,
double    i_tz,
TL_T_REAL o_fsEl[9*9],
TL_T_REAL o_fsEr[9*9],
TL_T_REAL o_fsAl[6*9],
TL_T_REAL o_fsAr[6*9],
bool      i_freeSurface ) {
double l_tE[9][9];
double l_tA[6][6];
double l_tm1[9][9];

seismic::common::setupTrafo3d( i_nx, i_ny, i_nz,
i_sx, i_sy, i_sz,
i_tx, i_ty, i_tz,
l_tE );

for( unsigned short l_q0 = 0; l_q0 < 6; l_q0++ )
for( unsigned short l_q1 = 0; l_q1 < 6; l_q1++ )
l_tA[l_q0][l_q1] = l_tE[l_q0][l_q1];

seismic::common::setupTrafoInv3d( i_nx, i_ny, i_nz,
i_sx, i_sy, i_sz,
i_tx, i_ty, i_tz,
l_tm1 );

double l_fsE[2][9*9];
setups::Elasticity::fs( i_rhoL,
i_rhoR,
i_lamL,
i_lamR,
i_muL,
i_muR,
l_tE,
l_tm1,
l_fsE[0],
l_fsE[1],
i_freeSurface );
for( unsigned short l_en = 0; l_en < 9*9; l_en++ ) {
o_fsEl[l_en] = l_fsE[0][l_en];
o_fsEr[l_en] = l_fsE[1][l_en];
}

if( o_fsAl != nullptr && o_fsAr != nullptr ) {
double l_fsA[2][6*9];
setups::ViscoElasticity::fs( i_rhoL,
i_rhoR,
i_lamL,
i_lamR,
i_muL,
i_muR,
l_tA,
l_tm1,
l_fsA[0],
l_fsA[1],
i_freeSurface );

for( unsigned short l_en = 0; l_en < 6*9; l_en++ ) {
o_fsAl[l_en] = l_fsA[0][l_en];
o_fsAr[l_en] = l_fsA[1][l_en];
}
}
}


template< typename TL_T_LID,
typename TL_T_REAL >
static void initFs( TL_T_LID                i_nElsIn,
TL_T_LID                i_nElsSe,
TL_T_LID                i_nFas,
TL_T_LID                i_nCommElFa,
unsigned short const  * i_recvFa,
TL_T_LID       const  * i_recvEl,
TL_T_LID       const (* i_faEl)[2],
TL_T_LID       const (* i_elVe)[TL_N_VES_EL],
TL_T_LID       const (* i_elFa)[TL_N_FAS],
t_vertexChars  const  * i_veChars,
t_faceChars    const  * i_faChars,
t_elementChars const  * i_elChars,
t_bgPars       const  * i_bgPars,
t_bgPars       const  * i_bgParsRe,
TL_T_REAL            (* o_fsE[2])[TL_N_FAS][TL_N_ENS_FS_E],
TL_T_REAL            (* o_fsA[2])[TL_N_FAS][TL_N_ENS_FS_A] ) {
PP_INSTR_FUN("flux_solvers")

TL_T_LID l_nEls = i_nElsIn + i_nElsSe;

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_el = 0; l_el < l_nEls; l_el++ ) {
for( TL_T_LID l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
for( unsigned short l_sd = 0; l_sd < 2; l_sd++ ) {
for( unsigned short l_en = 0; l_en < TL_N_ENS_FS_E; l_en++ ) {
o_fsE[l_sd][l_el][l_fa][l_en] = std::numeric_limits< TL_T_REAL >::max();
}
if( o_fsA[0] != nullptr && o_fsA[1] != nullptr ) {
for( unsigned short l_en = 0; l_en < TL_N_ENS_FS_A; l_en++ ) {
o_fsA[l_sd][l_el][l_fa][l_en] = std::numeric_limits< TL_T_REAL >::max();
}
}
}
}
}

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_fa = 0; l_fa < i_nFas; l_fa++ ) {
TL_T_LID l_elL = i_faEl[l_fa][0];
TL_T_LID l_elR = i_faEl[l_fa][1];

bool l_exL = l_elL != std::numeric_limits< TL_T_LID >::max();
bool l_exR = l_elR != std::numeric_limits< TL_T_LID >::max();

unsigned short l_fIdL = std::numeric_limits< unsigned short >::max();
unsigned short l_fIdR = std::numeric_limits< unsigned short >::max();
for( unsigned short l_fi = 0; l_fi < TL_N_FAS; l_fi++ ) {
if( l_exL && i_elFa[l_elL][l_fi] == l_fa ) {
EDGE_CHECK_EQ( l_fIdL, std::numeric_limits< unsigned short >::max() );
l_fIdL = l_fi;
}
if( l_exR && i_elFa[l_elR][l_fi] == l_fa ) {
EDGE_CHECK_EQ( l_fIdR, std::numeric_limits <unsigned short >::max() );
l_fIdR = l_fi;
}
}

bool l_periodic = (i_faChars[l_fa].spType & PERIODIC) == PERIODIC;
EDGE_CHECK( !l_exL || l_fIdL != std::numeric_limits< unsigned short >::max() || l_periodic ) << l_fa;
EDGE_CHECK( !l_exR || l_fIdR != std::numeric_limits< unsigned short >::max() || l_periodic ) << l_fa;

TL_T_REAL l_rhoL = std::numeric_limits< TL_T_REAL >::max();
TL_T_REAL l_rhoR = std::numeric_limits< TL_T_REAL >::max();
TL_T_REAL l_lamL = std::numeric_limits< TL_T_REAL >::max();
TL_T_REAL l_lamR = std::numeric_limits< TL_T_REAL >::max();
TL_T_REAL l_muL = std::numeric_limits< TL_T_REAL >::max();
TL_T_REAL l_muR = std::numeric_limits< TL_T_REAL >::max();
if( l_exL ) {
l_rhoL = i_bgPars[l_elL].rho; l_lamL = i_bgPars[l_elL].lam; l_muL = i_bgPars[l_elL].mu;
}
else {
EDGE_LOG_FATAL;
}

if( l_exR ) {
l_rhoR = i_bgPars[l_elR].rho; l_lamR = i_bgPars[l_elR].lam; l_muR = i_bgPars[l_elR].mu;
}
else if( (!l_exR) && (    (i_faChars[l_fa].spType & OUTFLOW)      == OUTFLOW
|| (i_faChars[l_fa].spType & FREE_SURFACE) == FREE_SURFACE ) ) {
l_rhoR = i_bgPars[l_elL].rho; l_lamR = i_bgPars[l_elL].lam; l_muR = i_bgPars[l_elL].mu;
}
else if( l_elL >= i_nElsIn ) {
bool l_reR = false;
for( std::size_t l_co = 0; l_co < i_nCommElFa; l_co++ ) {
if( i_recvEl[l_co] == l_elL && i_recvFa[l_co] == l_fIdL ) {
l_rhoR = i_bgParsRe[l_co].rho; l_lamR = i_bgParsRe[l_co].lam; l_muR = i_bgParsRe[l_co].mu;
l_reR = true;
break;
}
}
EDGE_CHECK( l_reR );
}
else {
EDGE_LOG_FATAL;
}

l_exL = l_exL && l_fIdL != std::numeric_limits< unsigned short >::max();
l_exR = l_exL && l_fIdR != std::numeric_limits< unsigned short >::max();

if( l_exL ) {
if( TL_N_DIS == 2 ) {
setUpFs( l_rhoL, l_rhoR,
l_lamL, l_lamR,
l_muL,  l_muR,
i_faChars[l_fa].outNormal[0],
i_faChars[l_fa].outNormal[1],
o_fsE[0][l_elL][l_fIdL],
o_fsE[1][l_elL][l_fIdL],
(o_fsA[0] != nullptr) ? o_fsA[0][l_elL][l_fIdL] : nullptr,
(o_fsA[1] != nullptr) ? o_fsA[1][l_elL][l_fIdL] : nullptr,
(i_faChars[l_fa].spType & FREE_SURFACE) == FREE_SURFACE );
}
else if( TL_N_DIS == 3 ) {
setUpFs( l_rhoL, l_rhoR,
l_lamL, l_lamR,
l_muL,  l_muR,
i_faChars[l_fa].outNormal[0],
i_faChars[l_fa].outNormal[1],
i_faChars[l_fa].outNormal[2],
i_faChars[l_fa].tangent0[0],
i_faChars[l_fa].tangent0[1],
i_faChars[l_fa].tangent0[2],
i_faChars[l_fa].tangent1[0],
i_faChars[l_fa].tangent1[1],
i_faChars[l_fa].tangent1[2],
o_fsE[0][l_elL][l_fIdL],
o_fsE[1][l_elL][l_fIdL],
(o_fsA[0] != nullptr) ? o_fsA[0][l_elL][l_fIdL] : nullptr,
(o_fsA[1] != nullptr) ? o_fsA[1][l_elL][l_fIdL] : nullptr,
(i_faChars[l_fa].spType & FREE_SURFACE) == FREE_SURFACE );
}
}

if( l_exR ) {
EDGE_CHECK_NE( (i_faChars[l_fa].spType & FREE_SURFACE), FREE_SURFACE );

if( TL_N_DIS == 2 ) {
setUpFs(  l_rhoR, l_rhoL,
l_lamR, l_lamL,
l_muR,  l_muL,
-i_faChars[l_fa].outNormal[0],
-i_faChars[l_fa].outNormal[1],
o_fsE[0][l_elR][l_fIdR],
o_fsE[1][l_elR][l_fIdR],
(o_fsA[0] != nullptr) ? o_fsA[0][l_elR][l_fIdR] : nullptr,
(o_fsA[1] != nullptr) ? o_fsA[1][l_elR][l_fIdR] : nullptr,
false );
}
else if( TL_N_DIS == 3 ) {
setUpFs(  l_rhoR, l_rhoL,
l_lamR, l_lamL,
l_muR,  l_muL,
-i_faChars[l_fa].outNormal[0],
-i_faChars[l_fa].outNormal[1],
-i_faChars[l_fa].outNormal[2],
i_faChars[l_fa].tangent0[0],
i_faChars[l_fa].tangent0[1],
i_faChars[l_fa].tangent0[2],
i_faChars[l_fa].tangent1[0],
i_faChars[l_fa].tangent1[1],
i_faChars[l_fa].tangent1[2],
o_fsE[0][l_elR][l_fIdR],
o_fsE[1][l_elR][l_fIdR],
(o_fsA[0] != nullptr) ? o_fsA[0][l_elR][l_fIdR] : nullptr,
(o_fsA[1] != nullptr) ? o_fsA[1][l_elR][l_fIdR] : nullptr,
false );
}
}

TL_T_REAL l_veCrds[2][TL_N_DIS][TL_N_VES_EL];
if( l_exL ) {
mesh::common< TL_T_EL >::getElVeCrds( l_elL,
i_elVe,
i_veChars,
l_veCrds[0] );
}
if( l_exR ) {
mesh::common< TL_T_EL >::getElVeCrds( l_elR,
i_elVe,
i_veChars,
l_veCrds[1] );
}

TL_T_REAL l_jDet[2] = { std::numeric_limits< TL_T_REAL >::max(), std::numeric_limits< TL_T_REAL >::max() };

TL_T_REAL l_jac[TL_N_DIS][TL_N_DIS];
if( l_exL ) {
linalg::Mappings::evalJac( TL_T_EL, l_veCrds[0][0], l_jac[0] );
l_jDet[0] = linalg::Matrix::det( l_jac );
}
if( l_exR ) {
linalg::Mappings::evalJac( TL_T_EL, l_veCrds[1][0], l_jac[0] );
l_jDet[1] = linalg::Matrix::det( l_jac );
}

EDGE_CHECK( !l_exL || l_jDet[0] > 0 );
EDGE_CHECK( !l_exR || l_jDet[1] > 0 );
EDGE_CHECK( i_faChars[l_fa].area > 0 );

TL_T_REAL l_sca[2] = { std::numeric_limits< TL_T_REAL >::max(), std::numeric_limits< TL_T_REAL >::max() };
if( l_exL ) {
l_sca[0] = -i_faChars[l_fa].area / l_jDet[0];
if( TL_T_EL == TET4 ) l_sca[0] *= 2;
}
if( l_exR ) {
l_sca[1] = -i_faChars[l_fa].area / l_jDet[1];
if( TL_T_EL == TET4 ) l_sca[1] *= 2;
}

for( unsigned short l_en = 0; l_en < TL_N_ENS_FS_E; l_en++ ) {
EDGE_CHECK( !l_exL || o_fsE[0][l_elL][l_fIdL][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exR || o_fsE[0][l_elR][l_fIdR][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exL || o_fsE[1][l_elL][l_fIdL][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exR || o_fsE[1][l_elR][l_fIdR][l_en] != std::numeric_limits< TL_T_REAL >::max() );

if( l_exL ) o_fsE[0][l_elL][l_fIdL][l_en] *= l_sca[0];
if( l_exL ) o_fsE[1][l_elL][l_fIdL][l_en] *= l_sca[0];

if( l_exR ) o_fsE[0][l_elR][l_fIdR][l_en] *= l_sca[1];
if( l_exR ) o_fsE[1][l_elR][l_fIdR][l_en] *= l_sca[1];
}

if( o_fsA[0] != nullptr && o_fsA[1] != nullptr ) {
for( unsigned short l_en = 0; l_en < TL_N_ENS_FS_A; l_en++ ) {
EDGE_CHECK( !l_exL || o_fsA[0][l_elL][l_fIdL][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exR || o_fsA[0][l_elR][l_fIdR][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exL || o_fsA[1][l_elL][l_fIdL][l_en] != std::numeric_limits< TL_T_REAL >::max() );
EDGE_CHECK( !l_exR || o_fsA[1][l_elR][l_fIdR][l_en] != std::numeric_limits< TL_T_REAL >::max() );

if( l_exL ) o_fsA[0][l_elL][l_fIdL][l_en] *= l_sca[0];
if( l_exL ) o_fsA[1][l_elL][l_fIdL][l_en] *= l_sca[0];

if( l_exR ) o_fsA[0][l_elR][l_fIdR][l_en] *= l_sca[1];
if( l_exR ) o_fsA[1][l_elR][l_fIdR][l_en] *= l_sca[1];
}
}
}
}

};

#endif