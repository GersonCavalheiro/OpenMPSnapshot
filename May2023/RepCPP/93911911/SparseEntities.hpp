

#ifndef EDGE_DATA_SPARSE_ENTITIES_HPP
#define EDGE_DATA_SPARSE_ENTITIES_HPP

#include "parallel/Distributed.h"
#include <limits>
#include "io/logging.h"
#include "linalg/Geom.hpp"

#include "EntityLayout.h"
namespace edge {
namespace data {
class SparseEntities;
}
}


class edge::data::SparseEntities {
private:

template< typename TL_T_LAYOUT >
static void initPartLayout( TL_T_LAYOUT const & i_deLayout,
TL_T_LAYOUT       & o_spLayout ) {
o_spLayout.timeGroups.resize( i_deLayout.timeGroups.size() );
for( std::size_t l_tg = 0; l_tg < o_spLayout.timeGroups.size(); l_tg++ ) {
o_spLayout.timeGroups[l_tg].inner.size  = 0;

o_spLayout.timeGroups[l_tg].send.resize(    i_deLayout.timeGroups[l_tg].send.size()    );
o_spLayout.timeGroups[l_tg].receive.resize( i_deLayout.timeGroups[l_tg].receive.size() );

o_spLayout.timeGroups[l_tg].neRanks.resize( i_deLayout.timeGroups[l_tg].neRanks.size() );
o_spLayout.timeGroups[l_tg].neTgs.resize(   i_deLayout.timeGroups[l_tg].neTgs.size() );
}
}

public:

template< typename TL_T_INT_LID, typename TL_T_INT_SP, typename TL_T_EN_CHARS >
static TL_T_INT_LID nSp( TL_T_INT_LID          i_nEn,
TL_T_INT_SP           i_spType,
TL_T_EN_CHARS const * i_chars ) {
TL_T_INT_LID l_nSp = 0;

for( TL_T_INT_LID l_en = 0; l_en < i_nEn; l_en++ )
if( (i_chars[l_en].spType & i_spType) == i_spType ) l_nSp++;

return l_nSp;
}


template <typename TL_T_INT_SP, typename TL_T_EN_CHARS>
static void denseToSparse( unsigned short        i_nTgs,
TL_T_INT_SP           i_spType,
TL_T_EN_CHARS const * i_chars,
std::size_t   const * i_nElsInDe,
std::size_t   const * i_nElsSeDe,
std::size_t         * o_nElsInSp,
std::size_t         * o_nElsSeSp ) {
std::size_t l_first = 0;
for( unsigned short l_tg = 0; l_tg < i_nTgs; l_tg++ ) {
o_nElsInSp[l_tg] = 0;

for( std::size_t l_el = l_first; l_el < l_first+i_nElsInDe[l_tg]; l_el++ ) {
if( (i_chars[l_el].spType & i_spType) == i_spType ) o_nElsInSp[l_tg]++;
}
l_first += i_nElsInDe[l_tg];
}
for( unsigned short l_tg = 0; l_tg < i_nTgs; l_tg++ ) {
o_nElsSeSp[l_tg] = 0;

for( std::size_t l_el = l_first; l_el < l_first+i_nElsSeDe[l_tg]; l_el++ ) {
if( (i_chars[l_el].spType & i_spType) == i_spType ) o_nElsSeSp[l_tg]++;
}
l_first += i_nElsSeDe[l_tg];
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_DE_CHARS >
static void subRgnsSpId( TL_T_INT_LID         i_rgnFirst,
unsigned int         i_nSubRgns,
TL_T_INT_LID const * i_subRgnSizes,
TL_T_INT_SP          i_spType,
TL_T_DE_CHARS        i_deChars,
TL_T_INT_LID       * o_subRgnsFirstSp,
TL_T_INT_LID       * o_subRgnsSizeSp ) {
TL_T_INT_LID l_first = 0;
for( TL_T_INT_LID l_en = 0; l_en < i_rgnFirst; l_en++ ) {
if( (i_deChars[l_en].spType & i_spType) == i_spType ) l_first++;
}

TL_T_INT_LID l_en = i_rgnFirst;
for( unsigned int l_sr = 0; l_sr < i_nSubRgns; l_sr++ ) {
o_subRgnsFirstSp[l_sr] = l_first;

TL_T_INT_LID l_up = l_en +  i_subRgnSizes[l_sr];
o_subRgnsSizeSp[l_sr] = 0;

for( ; l_en < l_up; l_en++ ) {
if( (i_deChars[l_en].spType & i_spType) == i_spType ) {
l_first++;
o_subRgnsSizeSp[l_sr]++;
}
}
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_CHARS >
static void inherit( TL_T_INT_LID  i_nEn,
TL_T_INT_SP   i_spTypeIn,
TL_T_INT_SP   i_spTypeOut,
TL_T_CHARS   *io_chars ) {
for( TL_T_INT_LID l_en = 0; l_en < i_nEn; l_en++ ) {
if( (io_chars[l_en].spType & i_spTypeIn) == i_spTypeIn ) {
io_chars[l_en].spType |= i_spTypeOut;
}
}
}


template< typename TL_T_LID,
typename TL_T_INT_SP,
typename TL_T_EN0_CHARS,
typename TL_T_EN1_CHARS >
static void propAdj( TL_T_LID               i_nEn0,
unsigned short         i_nEn0PerEn1,
TL_T_LID       const * i_en0En1,
TL_T_INT_SP            i_spTypeIn,
TL_T_INT_SP            i_spTypeOut,
TL_T_EN0_CHARS const * i_charsEn0,
TL_T_EN1_CHARS       * o_charsEn1 ) {
TL_T_LID l_nEn1 = 0;

for( TL_T_LID l_en = 0; l_en < i_nEn0; l_en++ ) {
if( (i_charsEn0[l_en].spType & i_spTypeIn) == i_spTypeIn ) {
for( unsigned short l_ae = 0; l_ae < i_nEn0PerEn1; l_ae++ ) {
TL_T_LID l_aeId = i_en0En1[ l_en*i_nEn0PerEn1 + l_ae ];

if( l_aeId != std::numeric_limits< TL_T_LID >::max() ) {
l_nEn1 = std::max( l_nEn1, l_aeId+1 );
o_charsEn1[l_aeId].spType |= i_spTypeOut;
}
}
}
}
}


template< typename TL_T_LID,
typename TL_T_INT_SP,
typename TL_T_EN0_CHARS,
typename TL_T_EN1_CHARS >
static void propAdj( TL_T_LID                   i_nEn0,
TL_T_LID   const * const * i_en0En1,
TL_T_INT_SP                    i_spTypeIn,
TL_T_INT_SP                    i_spTypeOut,
TL_T_EN0_CHARS         const * i_charsEn0,
TL_T_EN1_CHARS               * o_charsEn1 ) {
TL_T_LID l_nEn1 = 0;

for( TL_T_LID l_en = 0; l_en < i_nEn0; l_en++ ) {
if( (i_charsEn0[l_en].spType & i_spTypeIn) == i_spTypeIn ) {
for( unsigned short l_ae = 0; l_ae < i_en0En1[l_en+1]-i_en0En1[l_en]; l_ae++ ) {
TL_T_LID l_aeId = i_en0En1[l_en][l_ae];

if( l_aeId != std::numeric_limits< TL_T_LID >::max() ) {
o_charsEn1[l_aeId].spType |= i_spTypeOut;
l_nEn1 = std::max( l_nEn1, l_aeId+1 );
}
}
}
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_CHARS >
static void linkSpDe( TL_T_INT_LID       i_nDe,
TL_T_INT_SP        i_spType,
TL_T_CHARS const * i_chars,
TL_T_INT_LID     * o_spDe ) {
TL_T_INT_LID l_sp = 0;

for( TL_T_INT_LID l_de = 0; l_de < i_nDe; l_de++ ) {
if( (i_chars[l_de].spType & i_spType) == i_spType ) {
o_spDe[l_sp] = l_de;
l_sp++;
}
}
}


template< typename TL_T_LID,
typename TL_T_SP,
typename TL_T_CHARS >
static void linkSpSp( TL_T_LID           i_nDe,
TL_T_SP            i_spType0,
TL_T_SP            i_spType1,
TL_T_CHARS const * i_chars,
TL_T_LID         * o_sp0Sp1 ) {
TL_T_LID l_sp0 = 0;
TL_T_LID l_sp1 = 0;

for( TL_T_LID l_de = 0; l_de < i_nDe; l_de++ ) {
if( (i_chars[l_de].spType & i_spType0) == i_spType0 ) {
if( (i_chars[l_de].spType & i_spType1) == i_spType1 )
o_sp0Sp1[l_sp0] = l_sp1;
else
o_sp0Sp1[l_sp0] = std::numeric_limits< TL_T_LID >::max();

l_sp0++;
}

if( (i_chars[l_de].spType & i_spType1) == i_spType1 )
l_sp1++;
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_ADJ_CHARS >
static void linkSpAdjSst( TL_T_INT_LID           i_nEn,
unsigned short         i_nAdjPerEn,
TL_T_INT_LID   const * i_enEn,
TL_T_INT_SP            i_spType,
TL_T_ADJ_CHARS const * i_charsAdj,
TL_T_INT_LID         * o_spLink ) {
if( i_nEn == 0 ) return;
EDGE_CHECK( i_nAdjPerEn > 0 );

TL_T_INT_LID l_nAdjEn = 0;
for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
for( unsigned short l_ae = 0; l_ae < i_nAdjPerEn; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[ l_de * i_nAdjPerEn + l_ae ];

if( l_aeId == std::numeric_limits< TL_T_INT_LID >::max() ) continue;
l_nAdjEn = std::max( l_nAdjEn, l_aeId );
}
}

l_nAdjEn++;

std::vector< TL_T_INT_LID > l_adjDeToSp;
l_adjDeToSp.resize( l_nAdjEn );
TL_T_INT_LID l_spId = 0;
for( TL_T_INT_LID l_de = 0; l_de < l_nAdjEn; l_de++ ) {
if( ( i_charsAdj[l_de].spType & i_spType ) == i_spType ) {
l_adjDeToSp[l_de] = l_spId;
l_spId++;
}
else l_adjDeToSp[l_de] = std::numeric_limits< TL_T_INT_LID >::max();
}

l_spId = 0;
for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
bool l_sp = false;
for( unsigned short l_ae = 0; l_ae < i_nAdjPerEn; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[ l_de * i_nAdjPerEn + l_ae ];

if( l_aeId == std::numeric_limits< TL_T_INT_LID >::max() ) continue;

if( ( i_charsAdj[l_aeId ].spType & i_spType ) == i_spType ) l_sp = true;
}

if( l_sp ) {
for( unsigned short l_ae = 0; l_ae < i_nAdjPerEn; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[ l_de * i_nAdjPerEn + l_ae ];

if( l_aeId == std::numeric_limits< TL_T_INT_LID >::max() ) continue;

if( ( i_charsAdj[l_aeId].spType & i_spType ) == i_spType ) o_spLink[ l_spId*i_nAdjPerEn + l_ae ] = l_adjDeToSp[l_aeId];
else                                                       o_spLink[ l_spId*i_nAdjPerEn + l_ae ] = std::numeric_limits< TL_T_INT_LID >::max();
}
l_spId++;
}
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_CHARS_FROM,
typename TL_T_CHARS_TO >
static void linkSpAdjDst( TL_T_INT_LID            i_nEn,
unsigned short          i_nAdjPerEn,
TL_T_INT_LID    const * i_enEn,
TL_T_INT_SP             i_spTypeFrom,
TL_T_INT_SP             i_spTypeTo,
TL_T_CHARS_FROM const * i_charsFrom,
TL_T_CHARS_TO   const * i_charsTo,
TL_T_INT_LID          * o_spLink ) {
if( i_nEn == 0 ) return;
EDGE_CHECK( i_nAdjPerEn > 0 );

TL_T_INT_LID l_nAdjEn = 0;
for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
for( unsigned short l_ae = 0; l_ae < i_nAdjPerEn; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[ l_de * i_nAdjPerEn + l_ae ];

if( l_aeId == std::numeric_limits< TL_T_INT_LID >::max() ) continue;

l_nAdjEn = std::max( l_nAdjEn, l_aeId );
}
}

l_nAdjEn++;

std::vector< TL_T_INT_LID > l_adjDeToSp;
l_adjDeToSp.resize( l_nAdjEn );
TL_T_INT_LID l_spId = 0;
for( TL_T_INT_LID l_de = 0; l_de < l_nAdjEn; l_de++ ) {
if( ( i_charsTo[l_de].spType & i_spTypeTo ) == i_spTypeTo ) {
l_adjDeToSp[l_de] = l_spId;
l_spId++;
}
else l_adjDeToSp[l_de] = std::numeric_limits< TL_T_INT_LID >::max();
}

l_spId = 0;
for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
if( (i_charsFrom[l_de].spType & i_spTypeFrom) == i_spTypeFrom ) {
for( unsigned short l_ae = 0; l_ae < i_nAdjPerEn; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[ l_de * i_nAdjPerEn + l_ae ];

if( l_aeId == std::numeric_limits< TL_T_INT_LID >::max() ) {
o_spLink[ l_spId*i_nAdjPerEn + l_ae ] = std::numeric_limits< TL_T_INT_LID >::max();
continue;
}

if( ( i_charsTo[l_aeId].spType & i_spTypeTo ) == i_spTypeTo ) o_spLink[ l_spId*i_nAdjPerEn + l_ae ] = l_adjDeToSp[l_aeId];
else                                                          o_spLink[ l_spId*i_nAdjPerEn + l_ae ] = std::numeric_limits< TL_T_INT_LID >::max();
}

l_spId++;
}
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_CHARS_FROM,
typename TL_T_CHARS_TO >
static void nLinkSpAdjDst( TL_T_INT_LID                    i_nEn,
TL_T_INT_LID    const * const * i_enEn,
TL_T_INT_SP                     i_spTypeFrom,
TL_T_INT_SP                     i_spTypeTo,
TL_T_CHARS_FROM         const * i_charsFrom,
TL_T_CHARS_TO           const * i_charsTo,
TL_T_INT_LID                  & o_nSpLinkRaw,
TL_T_INT_LID                  & o_nSpLinkPtr ) {
o_nSpLinkRaw = o_nSpLinkPtr = 0;

for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
if( (i_charsFrom[l_de].spType & i_spTypeFrom) != i_spTypeFrom ) continue;

o_nSpLinkPtr++;

for( unsigned short l_ae = 0; l_ae < i_enEn[l_de+1]-i_enEn[l_de]; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[l_de][l_ae];

EDGE_CHECK( l_aeId != std::numeric_limits< TL_T_INT_LID >::max() );

if( (i_charsTo[l_aeId].spType & i_spTypeTo) == i_spTypeTo ) o_nSpLinkRaw++;
}
}
}


template< typename TL_T_INT_LID,
typename TL_T_INT_SP,
typename TL_T_CHARS_FROM,
typename TL_T_CHARS_TO >
static void linkSpAdjDst( TL_T_INT_LID                    i_nEn,
TL_T_INT_LID    const * const * i_enEn,
TL_T_INT_SP                     i_spTypeFrom,
TL_T_INT_SP                     i_spTypeTo,
TL_T_CHARS_FROM         const * i_charsFrom,
TL_T_CHARS_TO           const * i_charsTo,
TL_T_INT_LID                  * o_spLinkRaw,
TL_T_INT_LID                 ** o_spLinkPtr ) {
o_spLinkPtr[0] = o_spLinkRaw;

TL_T_INT_LID l_nAdjEn = 0;
for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
for( unsigned short l_ae = 0; l_ae < i_enEn[l_de+1]-i_enEn[l_de]; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[l_de][l_ae];
l_nAdjEn = std::max( l_nAdjEn, l_aeId );
}
}

l_nAdjEn++;

std::vector< TL_T_INT_LID > l_adjDeToSp;
l_adjDeToSp.resize( l_nAdjEn );

TL_T_INT_LID l_spId = 0;
for( TL_T_INT_LID l_de = 0; l_de < l_nAdjEn; l_de++ ) {
if( (i_charsTo[l_de].spType & i_spTypeTo) == i_spTypeTo ) {
l_adjDeToSp[l_de] = l_spId;
l_spId++;
}
else l_adjDeToSp[l_de] = std::numeric_limits< TL_T_INT_LID >::max();
}


l_spId = 0;
TL_T_INT_LID * l_raw = o_spLinkRaw;

for( TL_T_INT_LID l_de = 0; l_de < i_nEn; l_de++ ) {
if( (i_charsFrom[l_de].spType & i_spTypeFrom) == i_spTypeFrom ) {
o_spLinkPtr[l_spId] = l_raw;

for( unsigned short l_ae = 0; l_ae < i_enEn[l_de+1]-i_enEn[l_de]; l_ae++ ) {
TL_T_INT_LID l_aeId = i_enEn[l_de][l_ae];

if( (i_charsTo[l_aeId].spType & i_spTypeTo) == i_spTypeTo ) {
*l_raw = l_adjDeToSp[l_aeId];
l_raw++;
}
}

l_spId++;
}
}

o_spLinkPtr[l_spId] = l_raw;
}


template< typename TL_T_LID,
typename TL_T_REAL,
typename TL_T_EN,
typename TL_T_CHARS_VE >
static TL_T_LID ptToEn( TL_T_EN                i_enType,
TL_T_LID               i_nPts,
TL_T_REAL     const (* i_ptCrds)[3],
TL_T_LID               i_nEns,
TL_T_LID      const  * i_enVe,
TL_T_CHARS_VE const  * i_charsVe,
TL_T_LID             * o_de ) {
unsigned short l_nVe = C_ENT[i_enType].N_VERTICES;

TL_T_REAL *l_minDist = new TL_T_REAL[i_nPts];

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < i_nPts; l_pt++ ) {
o_de[l_pt]      = std::numeric_limits< TL_T_LID >::max();
l_minDist[l_pt] = std::numeric_limits< TL_T_REAL >::max();
}

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_pt = 0; l_pt < i_nPts; l_pt++ ) {
for( TL_T_LID l_en = 0; l_en < i_nEns; l_en++ ) {
EDGE_CHECK_LE( l_nVe, 8 );
TL_T_REAL l_tmpVe[ 3*8 ];
for( unsigned short l_ve = 0; l_ve < l_nVe; l_ve++ ) {
TL_T_LID l_veId = i_enVe[l_en*l_nVe+l_ve];

for( unsigned short l_di = 0; l_di < 3; l_di++ ) {
l_tmpVe[l_di*l_nVe + l_ve] = i_charsVe[l_veId].coords[l_di];
}
}

TL_T_REAL l_tmpCrds[3];
for( unsigned short l_di = 0; l_di < 3; l_di++ ) {
l_tmpCrds[l_di] = i_ptCrds[l_pt][l_di];
}
edge::linalg::Geom::closestPoint( i_enType,
l_tmpVe,
l_tmpCrds );
TL_T_REAL l_dist = edge::linalg::GeomT< 3 >::norm( l_tmpCrds,
i_ptCrds[l_pt] );

if( l_dist < l_minDist[l_pt] ) {
o_de[l_pt] = l_en;
l_minDist[l_pt] = l_dist;
}
}
}

unsigned short *l_own = new unsigned short [i_nPts];
parallel::Distributed::min( i_nPts,
l_minDist,
l_own );

TL_T_LID l_nOwn= 0;
for( TL_T_LID l_pt = 0; l_pt < i_nPts; l_pt++ ) {
if( l_own[l_pt] != 1 )
o_de[l_pt] = std::numeric_limits< TL_T_LID >::max();
else
l_nOwn++;
}

delete[] l_own;
delete[] l_minDist;

return l_nOwn;
}
};
#endif
