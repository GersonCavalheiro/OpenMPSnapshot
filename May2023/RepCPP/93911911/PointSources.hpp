

#ifndef EDGE_SEISMIC_SETUPS_POINT_SOURCES_HPP
#define EDGE_SEISMIC_SETUPS_POINT_SOURCES_HPP

#include <string>
#include "hdf5.h"
#include "data/Dynamic.h"
#include "data/SparseEntities.hpp"
#include "mesh/common.hpp"
#include "dg/Basis.h"
#include "linalg/Series.hpp"
#include "io/logging.h"

namespace edge {
namespace seismic {
namespace setups {
template< typename       TL_T_REAL,
typename       TL_T_LID,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class PointSources;
}
}
}


template< typename       TL_T_REAL,
typename       TL_T_LID,
t_entityType   TL_T_EL,
unsigned short TL_O_SP,
unsigned short TL_N_CRS >
class edge::seismic::setups::PointSources {
private:
static unsigned short const TL_N_DIS = C_ENT[ TL_T_EL ].N_DIM;

static unsigned short const TL_N_VES = C_ENT[TL_T_EL].N_VERTICES;

static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_QTS_SIM = (TL_N_DIS == 2) ? 5 : 9;

static unsigned short const TL_N_QTS_SRC = TL_N_QTS_SIM;

typedef struct PointSourcesData {
TL_T_LID nPts = 0;

TL_T_LID *el = nullptr;

TL_T_REAL *times = nullptr;

TL_T_REAL *dts = nullptr;

TL_T_REAL (*bEvals)[TL_N_MDS] = nullptr;

TL_T_REAL (*scas)[TL_N_QTS_SRC] = nullptr;

TL_T_LID *tsPtrs = nullptr;

TL_T_REAL *tss = nullptr;
} t_ps;

t_ps m_psFused[TL_N_CRS];


TL_T_LID (*m_elSpPs)[TL_N_CRS] = nullptr;


static void getAcDu( TL_T_LID                        i_nPss,
TL_T_LID                const * i_psElIds,
std::vector< TL_T_LID >       & o_psIdsAc,
std::vector< TL_T_LID >       & o_psElIdsAc ) {
PP_INSTR_FUN("get_ac_du")

for( TL_T_LID l_ps = 0; l_ps < i_nPss; l_ps++ ) {
if( i_psElIds[l_ps] != std::numeric_limits< TL_T_LID >::max() ) {
TL_T_LID l_lId = i_psElIds[l_ps];

o_psIdsAc.push_back( l_ps  );
o_psElIdsAc.push_back( l_lId );
}
}
}


static void getSrcPerms( std::vector< TL_T_LID > const &i_elId,
std::vector< TL_T_LID >       &o_per,
std::vector< TL_T_LID >       &o_perI ) {
PP_INSTR_FUN("get_src_perms")

o_per.resize(  i_elId.size() );
o_perI.resize( i_elId.size() );
for( std::size_t l_el = 0; l_el < i_elId.size(); l_el++ ) {
o_per[l_el] = l_el;
o_perI[l_el] = l_el;
}
std::sort( o_per.begin(), o_per.end(),
[&](const int& a, const int& b) { return i_elId[a] < i_elId[b]; }
);
std::sort( o_perI.begin(), o_perI.end(),
[&](const int& a, const int& b) { return o_per[a] < o_per[b]; }
);
}


template< typename TL_T_CHARS_VE >
static void getIds( std::string             const  & i_path,
TL_T_LID                         i_nEls,
TL_T_LID                const (* i_elVe)[TL_N_VES],
TL_T_CHARS_VE           const  * i_charsVe,
std::vector< TL_T_LID >        & o_srcEls,
std::vector< TL_T_LID >        & o_srcIds ) {
PP_INSTR_FUN("get_ids")

herr_t l_h5Err;

hid_t l_h5 = H5Fopen( i_path.c_str(),
H5F_ACC_RDONLY,
H5P_DEFAULT );

hid_t l_ptsDset = H5Dopen( l_h5,
"points",
H5P_DEFAULT );

hid_t l_ptsTy = H5Dget_type(l_ptsDset);
EDGE_CHECK_EQ( H5Tget_class(l_ptsTy), H5T_FLOAT );

hid_t  l_ptsDsp = H5Dget_space( l_ptsDset );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_ptsDsp ),   2 );

hsize_t l_ptsDis[2];
H5Sget_simple_extent_dims( l_ptsDsp, l_ptsDis, NULL );
EDGE_CHECK_EQ( l_ptsDis[1], TL_N_DIS );

float (*l_ptsRaw)[TL_N_DIS] = new float[l_ptsDis[0]][TL_N_DIS];
l_h5Err = H5Dread( l_ptsDset,
H5T_NATIVE_FLOAT,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_ptsRaw );
EDGE_CHECK_GE( l_h5Err, 0 );

double (*l_ptsRaw3d)[3] = new double[l_ptsDis[0]][3];
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( hsize_t l_pt = 0; l_pt < l_ptsDis[0]; l_pt++ ) {
for( unsigned short l_di = 0; l_di < 3; l_di++ )
l_ptsRaw3d[l_pt][l_di] = 0;
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ )
l_ptsRaw3d[l_pt][l_di] = l_ptsRaw[l_pt][l_di];
}
delete[] l_ptsRaw;

std::vector< TL_T_LID > l_ptIds( l_ptsDis[0] );

edge::data::SparseEntities::ptToEn( TL_T_EL,
(TL_T_LID) l_ptsDis[0],
l_ptsRaw3d,
i_nEls,
i_elVe[0],
i_charsVe,
l_ptIds.data() );

delete[] l_ptsRaw3d;

getAcDu( l_ptsDis[0],
l_ptIds.data(),
o_srcIds,
o_srcEls );

l_h5Err = H5Sclose(l_ptsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Tclose(l_ptsTy);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Dclose(l_ptsDset);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Fclose(l_h5);
EDGE_CHECK_GE( l_h5Err, 0 );
}


template< typename TL_T_SP,
typename TL_T_CHARS_EL >
static void elSpPs( TL_T_LID                            i_nEl,
TL_T_SP                             i_spTy,
TL_T_CHARS_EL           const     * i_charsEl,
std::vector< TL_T_LID > const       i_srcElP[TL_N_CRS],
data::Dynamic                     & io_dynMem,
TL_T_LID                      ( * & o_elSpPs)[TL_N_CRS] ) {
TL_T_LID l_nElSrc = 0;

for( TL_T_LID l_el = 0; l_el < i_nEl; l_el++ ) {
if( (i_charsEl[l_el].spType & i_spTy) == i_spTy ) {
EDGE_CHECK( (i_charsEl[l_el].spType & LIMIT) != LIMIT )
<< "Sources in sub-cell limited elements are not supported.";
l_nElSrc++;
}
}

o_elSpPs =  (TL_T_LID (*)[TL_N_CRS]) io_dynMem.allocate(
sizeof(TL_T_LID) * TL_N_CRS * (l_nElSrc+1) );

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_elSpPs[0][l_cr] = 0;
}

TL_T_LID l_elSpPs = 0;

TL_T_LID l_ptSrc[TL_N_CRS];
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) l_ptSrc[l_cr] = 0;

for( TL_T_LID l_el = 0; l_el < i_nEl; l_el++ ) {
if( (i_charsEl[l_el].spType & i_spTy) == i_spTy ) {
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_elSpPs[l_elSpPs][l_cr] = l_ptSrc[l_cr];

for( std::size_t l_ps = l_ptSrc[l_cr]; l_ps < i_srcElP[l_cr].size(); l_ps++ ) {
EDGE_CHECK_LE( l_el, i_srcElP[l_cr][l_ps] );

if( i_srcElP[l_cr][l_ps] == l_el ) l_ptSrc[l_cr]++;
else                               break;
}
}

l_elSpPs++;
}
}

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
o_elSpPs[l_nElSrc][l_cr] = l_ptSrc[l_cr];
}
}


template< typename TL_T_CHARS_VE >
static void initBasisEvals( std::string             const  & i_path,
std::vector< TL_T_LID > const  & i_srcElsP,
std::vector< TL_T_LID > const  & i_srcIdsP,
TL_T_LID                const (* i_elVe)[TL_N_VES],
TL_T_CHARS_VE           const  * i_charsVe,
TL_T_REAL               const    i_massI[TL_N_MDS],
edge::data::Dynamic            & io_dynMem,
t_ps                           & o_ps ) {
herr_t l_h5Err;

hid_t l_h5 = H5Fopen( i_path.c_str(),
H5F_ACC_RDONLY,
H5P_DEFAULT );

hid_t l_ptsDset = H5Dopen( l_h5,
"points",
H5P_DEFAULT );

hid_t l_ptsTy = H5Dget_type(l_ptsDset);
EDGE_CHECK_EQ( H5Tget_class(l_ptsTy), H5T_FLOAT );

hid_t  l_ptsDsp = H5Dget_space( l_ptsDset );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_ptsDsp ), 2 );

hsize_t l_ptsDis[2];
H5Sget_simple_extent_dims( l_ptsDsp, l_ptsDis, NULL );
EDGE_CHECK_EQ( l_ptsDis[1], TL_N_DIS );

std::size_t l_bEvalsSize = i_srcIdsP.size() * TL_N_MDS * sizeof(TL_T_REAL);
o_ps.bEvals = (TL_T_REAL (*)[TL_N_MDS]) io_dynMem.allocate( l_bEvalsSize );

float (*l_ptsRaw)[TL_N_DIS] = new float[l_ptsDis[0]][TL_N_DIS];
l_h5Err = H5Dread( l_ptsDset,
H5T_NATIVE_FLOAT,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_ptsRaw );
EDGE_CHECK_GE( l_h5Err, 0 );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_pt = 0; l_pt < i_srcIdsP.size(); l_pt++ ) {
double l_veCrds[TL_N_DIS][TL_N_VES];
mesh::common< TL_T_EL >::getElVeCrds( i_srcElsP[l_pt],
i_elVe,
i_charsVe,
l_veCrds );

TL_T_LID l_ptId = i_srcIdsP[l_pt];
double l_ptCrds[TL_N_DIS];
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) l_ptCrds[l_di] = l_ptsRaw[l_ptId][l_di];

edge::linalg::Geom::closestPoint( TL_T_EL,
l_veCrds[0],
l_ptCrds );

double l_refCrds[3] = {0,0,0};
edge::linalg::Mappings::phyToRef( TL_T_EL,
l_veCrds[0],
l_ptCrds,
l_refCrds );

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
EDGE_CHECK_GT( l_refCrds[l_di], -TOL.MESH );
EDGE_CHECK_LT( l_refCrds[l_di], 1+TOL.MESH );
}

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_ps.bEvals[l_pt][l_md] = dg::Basis::evalBasis( l_md,
TL_T_EL,
l_refCrds[0],
l_refCrds[1],
l_refCrds[2] );
}

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ ) {
o_ps.bEvals[l_pt][l_md] *= i_massI[l_md];
}

double l_jac[TL_N_DIS][TL_N_DIS];
edge::linalg::Mappings::evalJac( TL_T_EL,
l_veCrds[0],
l_jac[0] );

double l_jacDet = TL_T_REAL(1) / edge::linalg::Matrix::det( TL_N_DIS, l_jac[0] );

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
o_ps.bEvals[l_pt][l_md] *= l_jacDet;
}
delete[] l_ptsRaw;

l_h5Err = H5Sclose(l_ptsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Tclose(l_ptsTy);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Dclose(l_ptsDset);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Fclose(l_h5);
EDGE_CHECK_GE( l_h5Err, 0 );
}


template< typename TL_T_CHARS_VE >
static void initTsData( std::string             const  & i_path,
std::vector< TL_T_LID > const  & i_srcElsP,
std::vector< TL_T_LID > const  & i_srcIdsP,
TL_T_LID                const (* i_elVe)[TL_N_VES],
TL_T_CHARS_VE           const  * i_charsVe,
edge::data::Dynamic            & io_dynMem,
t_ps                           & o_ps ) {
o_ps.nPts = i_srcIdsP.size();

std::size_t l_pfElSize = o_ps.nPts * sizeof(TL_T_LID);
o_ps.el = (TL_T_LID*) io_dynMem.allocate( l_pfElSize );
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < o_ps.nPts; l_pt++ )
o_ps.el[l_pt] = i_srcElsP[l_pt];


herr_t l_h5Err;

hid_t l_h5 = H5Fopen( i_path.c_str(),
H5F_ACC_RDONLY,
H5P_DEFAULT );

hid_t l_ptsDset   = H5Dopen( l_h5,
"points",
H5P_DEFAULT );
hid_t l_scasDset  = H5Dopen( l_h5,
"scalings",
H5P_DEFAULT );
hid_t l_tParsDset = H5Dopen( l_h5,
"time_parameters",
H5P_DEFAULT );
hid_t l_tPtrsDset = H5Dopen( l_h5,
"time_pointers",
H5P_DEFAULT );
hid_t l_tSrsDset  = H5Dopen( l_h5,
"time_series",
H5P_DEFAULT );

hid_t l_ptsTy = H5Dget_type(l_ptsDset);
EDGE_CHECK_EQ( H5Tget_class(l_ptsTy), H5T_FLOAT );

hid_t l_scasTy = H5Dget_type(l_scasDset);
EDGE_CHECK_EQ( H5Tget_class(l_scasTy), H5T_FLOAT );

hid_t l_tParsTy = H5Dget_type(l_tParsDset);
EDGE_CHECK_EQ( H5Tget_class(l_tParsTy), H5T_FLOAT );

hid_t l_tPtrsTy = H5Dget_type(l_tPtrsDset);
EDGE_CHECK_EQ( H5Tget_class(l_tPtrsTy), H5T_INTEGER );

hid_t l_tSrsTy = H5Dget_type(l_tSrsDset);
EDGE_CHECK_EQ( H5Tget_class(l_tSrsTy), H5T_FLOAT );

hid_t l_ptsDsp   = H5Dget_space( l_ptsDset );
hid_t l_scasDsp  = H5Dget_space( l_scasDset );
hid_t l_tParsDsp = H5Dget_space( l_tParsDset );
hid_t l_tPtrsDsp = H5Dget_space( l_tPtrsDset );
hid_t l_tSrsDsp  = H5Dget_space( l_tSrsDset );

EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_ptsDsp ),   2 );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_scasDsp ),  2 );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_tParsDsp ), 2 );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_tPtrsDsp ), 1 );
EDGE_CHECK_EQ( H5Sget_simple_extent_ndims( l_tSrsDsp ),  1 );

hsize_t l_ptsDis[2];
H5Sget_simple_extent_dims( l_ptsDsp, l_ptsDis, NULL );
hsize_t l_scasDis[2];
H5Sget_simple_extent_dims( l_scasDsp, l_scasDis, NULL );
hsize_t l_tParsDis[2];
H5Sget_simple_extent_dims( l_tParsDsp, l_tParsDis, NULL );
hsize_t l_tPtrsDis;
H5Sget_simple_extent_dims( l_tPtrsDsp, &l_tPtrsDis, NULL );
hsize_t l_tSrsDis;
H5Sget_simple_extent_dims( l_tSrsDsp, &l_tSrsDis, NULL );

EDGE_CHECK_EQ( l_ptsDis[1], TL_N_DIS );
EDGE_CHECK_EQ( l_scasDis[1], TL_N_QTS_SRC );
EDGE_CHECK_EQ( l_tParsDis[1], 2 );

EDGE_CHECK_EQ( l_ptsDis[0],   l_scasDis[0]  );
EDGE_CHECK_EQ( l_ptsDis[0],   l_tParsDis[0] );
EDGE_CHECK_EQ( l_ptsDis[0]+1, l_tPtrsDis    );

std::size_t l_timesSize = o_ps.nPts * sizeof(TL_T_REAL);
o_ps.times = (TL_T_REAL*) io_dynMem.allocate( l_timesSize );

std::size_t l_dtsSize = o_ps.nPts * sizeof(TL_T_REAL);
o_ps.dts = (TL_T_REAL*) io_dynMem.allocate( l_dtsSize );

float (*l_tParsRaw)[2] = new float[l_tParsDis[0]][2];
l_h5Err = H5Dread( l_tParsDset,
H5T_NATIVE_FLOAT,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_tParsRaw[0] );
EDGE_CHECK_GE( l_h5Err, 0 );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < o_ps.nPts; l_pt++ ) {
TL_T_LID l_ptId = i_srcIdsP[l_pt];
o_ps.times[l_pt] = l_tParsRaw[l_ptId][0];
o_ps.dts[l_pt]   = l_tParsRaw[l_ptId][1];
}

delete[] l_tParsRaw;

std::size_t l_scasSize = o_ps.nPts * TL_N_QTS_SRC * sizeof(TL_T_REAL);
o_ps.scas = (TL_T_REAL (*)[TL_N_QTS_SRC]) io_dynMem.allocate( l_scasSize );

float (*l_scasRaw)[TL_N_QTS_SRC] = new float[l_scasDis[0]][TL_N_QTS_SRC];
l_h5Err = H5Dread( l_scasDset,
H5T_NATIVE_FLOAT,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_scasRaw[0] );
EDGE_CHECK_GE( l_h5Err, 0 );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < o_ps.nPts; l_pt++ ) {
TL_T_LID l_ptId = i_srcIdsP[l_pt];
for( unsigned short l_qt = 0; l_qt < TL_N_QTS_SRC; l_qt++ ) {
o_ps.scas[l_pt][l_qt] = l_scasRaw[l_ptId][l_qt];
}
}

delete[] l_scasRaw;

std::size_t l_tsPtrsSize = (o_ps.nPts+1) * sizeof(TL_T_LID);
o_ps.tsPtrs = (TL_T_LID *) io_dynMem.allocate( l_tsPtrsSize );

uint64_t *l_tsPtsRaw = new uint64_t[l_tPtrsDis];
l_h5Err = H5Dread( l_tPtrsDset,
H5T_NATIVE_UINT64,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_tsPtsRaw );
EDGE_CHECK_GE( l_h5Err, 0 );

o_ps.tsPtrs[0] = 0;
for( TL_T_LID l_pt = 0; l_pt < o_ps.nPts; l_pt++ ) {
TL_T_LID l_ptId = i_srcIdsP[l_pt];

o_ps.tsPtrs[l_pt+1] = o_ps.tsPtrs[l_pt];
o_ps.tsPtrs[l_pt+1] += l_tsPtsRaw[l_ptId+1] - l_tsPtsRaw[l_ptId];
}

std::size_t l_tssSize = o_ps.tsPtrs[o_ps.nPts] * sizeof(TL_T_REAL);
o_ps.tss = (TL_T_REAL*) io_dynMem.allocate( l_tssSize );

float *l_tssRaw = new float[l_tSrsDis];
l_h5Err = H5Dread( l_tSrsDset,
H5T_NATIVE_FLOAT,
H5S_ALL,
H5S_ALL,
H5P_DEFAULT,
l_tssRaw );
EDGE_CHECK_GE( l_h5Err, 0 );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < o_ps.nPts; l_pt++ ) {
TL_T_LID l_ptId = i_srcIdsP[l_pt];

TL_T_LID l_off = o_ps.tsPtrs[l_pt];
TL_T_LID l_offRaw = l_tsPtsRaw[l_ptId];

TL_T_LID l_nSas = o_ps.tsPtrs[l_pt+1]-o_ps.tsPtrs[l_pt];

for( TL_T_LID l_sa = 0; l_sa < l_nSas; l_sa++ ) {
o_ps.tss[l_off+l_sa] = l_tssRaw[l_offRaw+l_sa];
}
}

delete[] l_tsPtsRaw;
delete[] l_tssRaw;

l_h5Err = H5Sclose(l_ptsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Sclose(l_scasDsp);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Sclose(l_tParsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Sclose(l_tPtrsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Sclose(l_tSrsDsp);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Tclose(l_ptsTy);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Tclose(l_scasTy);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Tclose(l_tParsTy);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Tclose(l_tPtrsTy);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Tclose(l_tSrsTy);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Dclose(l_ptsDset);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Dclose(l_scasDset);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Dclose(l_tParsDset);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Dclose(l_tPtrsDset);
EDGE_CHECK_GE( l_h5Err, 0 );
l_h5Err = H5Dclose(l_tSrsDset);
EDGE_CHECK_GE( l_h5Err, 0 );

l_h5Err = H5Fclose(l_h5);
EDGE_CHECK_GE( l_h5Err, 0 );
}

public:

template< typename TL_T_SP,
typename TL_T_CHARS_VE,
typename TL_T_CHARS_EL >
bool init( std::string         const    i_paths[TL_N_CRS],
TL_T_SP                      i_spTy,
TL_T_LID                     i_nEls,
TL_T_LID            const (* i_elVe)[TL_N_VES],
TL_T_CHARS_VE       const  * i_charsVe,
TL_T_REAL           const    i_massI[TL_N_MDS],
TL_T_CHARS_EL              * io_charsEl,
edge::data::Dynamic        & io_dynMem ) {
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
if( i_paths[l_cr] == "" ) return false;

std::vector< TL_T_LID > l_srcEls[TL_N_CRS];
std::vector< TL_T_LID > l_srcIds[TL_N_CRS];

std::vector< TL_T_LID > l_perm[TL_N_CRS];
std::vector< TL_T_LID > l_permI[TL_N_CRS];
std::vector< TL_T_LID > l_srcElsP[TL_N_CRS];
std::vector< TL_T_LID > l_srcIdsP[TL_N_CRS];

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
getIds( i_paths[l_cr],
i_nEls,
i_elVe,
i_charsVe,
l_srcEls[l_cr],
l_srcIds[l_cr] );
EDGE_CHECK_EQ( l_srcEls[l_cr].size(), l_srcIds[l_cr].size() );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_ps = 0; l_ps < l_srcEls[l_cr].size(); l_ps++ ) {
io_charsEl[ l_srcEls[l_cr][l_ps] ].spType |= i_spTy;
}

getSrcPerms( l_srcEls[l_cr], l_perm[l_cr], l_permI[l_cr] );

l_srcIdsP[l_cr].resize( l_perm[l_cr].size() );
l_srcElsP[l_cr].resize( l_perm[l_cr].size() );
for( std::size_t l_ps = 0; l_ps < l_srcIds[l_cr].size(); l_ps++ ) {
TL_T_LID l_psP = l_perm[l_cr][l_ps];
l_srcIdsP[l_cr][l_ps] = l_srcIds[l_cr][l_psP];
l_srcElsP[l_cr][l_ps] = l_srcEls[l_cr][l_psP];
}
}

elSpPs( i_nEls,
i_spTy,
io_charsEl,
l_srcElsP,
io_dynMem,
m_elSpPs );

for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
initBasisEvals( i_paths[l_cr],
l_srcElsP[l_cr],
l_srcIdsP[l_cr],
i_elVe,
i_charsVe,
i_massI,
io_dynMem,
m_psFused[l_cr] );

initTsData( i_paths[l_cr],
l_srcElsP[l_cr],
l_srcIdsP[l_cr],
i_elVe,
i_charsVe,
io_dynMem,
m_psFused[l_cr] );
}

return true;
}


void apply( TL_T_LID     i_first,
TL_T_LID     i_size,
TL_T_REAL    i_t1,
TL_T_REAL    i_t2,
TL_T_REAL (* io_dofs)[TL_N_QTS_SIM][TL_N_MDS][TL_N_CRS] ) const {
if( m_elSpPs != nullptr ) {}
else return;

for( TL_T_LID l_el = i_first; l_el < i_first+i_size; l_el++ ) {
for( unsigned short l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) {
for( TL_T_LID l_ps = m_elSpPs[l_el][l_cr]; l_ps < m_elSpPs[l_el+1][l_cr]; l_ps++ ) {
TL_T_REAL l_int[1] = { std::numeric_limits< TL_T_REAL >::max() };

TL_T_LID l_psFirst = m_psFused[l_cr].tsPtrs[l_ps];
TL_T_LID l_psSize  = m_psFused[l_cr].tsPtrs[l_ps+1]-l_psFirst;
linalg::Series< 1 >::integrate( m_psFused[l_cr].dts[l_ps],
m_psFused[l_cr].times[l_ps],
l_psSize,
(TL_T_REAL (*)[1]) m_psFused[l_cr].tss+l_psFirst,
i_t1,
i_t2,
l_int,
(TL_T_REAL) 0.0 );

for( unsigned short l_qt = 0; l_qt < TL_N_QTS_SIM; l_qt++ ) {
TL_T_LID l_elDe = m_psFused[l_cr].el[l_ps];

TL_T_REAL l_sca = l_int[0] * m_psFused[l_cr].scas[l_ps][l_qt];

for( unsigned short l_md = 0; l_md < TL_N_MDS; l_md++ )
io_dofs[l_elDe][l_qt][l_md][l_cr] += l_sca*m_psFused[l_cr].bEvals[l_ps][l_md];
}
}
}
}
}
};

#endif