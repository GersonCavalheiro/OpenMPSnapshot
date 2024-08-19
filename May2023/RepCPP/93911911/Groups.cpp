
#include "Groups.h"

#include "../io/logging.h"

void edge_v::time::Groups::getLoads( t_idx                  i_nEls,
double         const * i_tsCfl,
double         const * i_tsGroups,
unsigned short const * i_elTg,
double               & o_gts,
double               & o_ltsGrouped,
double               & o_ltsPerElement ) {
o_gts = i_nEls;

o_ltsGrouped = 0;
o_ltsPerElement = 0;

for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
unsigned short l_tg = i_elTg[l_el];
o_ltsGrouped += 1.0 / i_tsGroups[l_tg];
o_ltsPerElement += 1.0 / i_tsCfl[l_el];
}
}

edge_v::t_idx edge_v::time::Groups::normalizeElTgs( t_entityType     i_elTy,
t_idx            i_nEls,
t_idx    const * i_elFaEl,
unsigned short * io_elTg ) {
t_idx l_normAll = 0;

unsigned short l_nElFas = CE_N_FAS( i_elTy );

while( true ) {
t_idx l_normIt = 0;

for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
unsigned short l_tgEl = io_elTg[l_el];

for( unsigned short l_fa = 0; l_fa < l_nElFas; l_fa++ )  {
t_idx l_ad = i_elFaEl[l_el * l_nElFas + l_fa];

if( l_ad < std::numeric_limits< t_idx >::max() ) {
unsigned short l_tgAd = io_elTg[l_ad];
if( l_tgEl > l_tgAd+1 ) {
io_elTg[l_el] = l_tgAd+1;
l_normIt++;
l_normAll++;
}
}
}
}

if( l_normIt == 0 ) break;
}

return l_normAll;
}


void edge_v::time::Groups::setElTg( t_idx                  i_nEls,
unsigned short         i_nGroups,
double         const * i_tsGroups,
double         const * i_tsCfl,
unsigned short       * o_elTg ) {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
for( unsigned short l_tg = 0; l_tg < i_nGroups; l_tg++ ) {
if( i_tsCfl[l_el] < i_tsGroups[l_tg+1] ) {
o_elTg[l_el] = l_tg;
break;
}
}
}
}

void edge_v::time::Groups::nGroupEls( t_idx                  i_first,
t_idx                  i_nEls,
unsigned short         i_nGroups,
unsigned short const * i_elTg,
t_idx                * o_nGroupEls ) {
for( unsigned short l_tg = 0; l_tg < i_nGroups; l_tg++ )
o_nGroupEls[l_tg] = 0;

for( t_idx l_el = i_first; l_el < i_first+i_nEls; l_el++ ) {
unsigned short l_tg = i_elTg[l_el];
o_nGroupEls[l_tg]++;
}
}


edge_v::time::Groups::Groups( t_entityType           i_elTy,
t_idx                  i_nEls,
t_idx    const       * i_elFaEl,
unsigned short         i_nRates,
double         const * i_rates,
double                 i_funDt,
double         const * i_ts ) {
for( unsigned short l_ra = 0; l_ra < i_nRates; l_ra++ ) {
EDGE_V_CHECK_GT( i_rates[l_ra], 1.0 );
EDGE_V_CHECK_LE( i_rates[l_ra], 2.0 );
}
EDGE_V_CHECK_GT( i_funDt, 0 );
EDGE_V_CHECK_LE( i_funDt, 1 );

m_nEls = i_nEls;

m_nGroups = i_nRates+1;
m_tsIntervals = new double[ m_nGroups+1 ];
m_tsIntervals[0] = i_funDt;
m_tsIntervals[i_nRates+1] = std::numeric_limits< double >::max();
for( unsigned short l_ra = 0; l_ra < i_nRates; l_ra++ ) {
m_tsIntervals[l_ra+1] = m_tsIntervals[l_ra] * i_rates[l_ra];
}

m_elTg = new unsigned short[i_nEls];
setElTg( i_nEls,
m_nGroups,
m_tsIntervals,
i_ts,
m_elTg );

getLoads( m_nEls,
i_ts,
m_tsIntervals,
m_elTg,
m_loads[0],
m_loads[1],
m_loads[3] );

normalizeElTgs( i_elTy,
i_nEls,
i_elFaEl,
m_elTg );

m_nGroupEls = new t_idx[m_nGroups];
nGroupEls( 0,
i_nEls,
m_nGroups,
m_elTg,
m_nGroupEls );

getLoads( m_nEls,
i_ts,
m_tsIntervals,
m_elTg,
m_loads[0],
m_loads[2],
m_loads[3] );
}

edge_v::time::Groups::~Groups() {
delete[] m_tsIntervals;
delete[] m_elTg;
delete[] m_nGroupEls;
}

void edge_v::time::Groups::printStats() const {
EDGE_V_LOG_INFO << "time step histogram (group / #elements / range):";
for( unsigned short l_tg = 0; l_tg < m_nGroups; l_tg++ ) {
EDGE_V_LOG_INFO << "  " << l_tg
<< " " << m_nGroupEls[l_tg]
<< " [" << m_tsIntervals[l_tg] << ", " << m_tsIntervals[l_tg+1] << "[";
}

EDGE_V_LOG_INFO << "theoretical LTS speedups:";
EDGE_V_LOG_INFO << "  grouped              over GTS:                  " << m_loads[0] / m_loads[1];
EDGE_V_LOG_INFO << "  grouped (normalized) over GTS:                  " << m_loads[0] / m_loads[2];
EDGE_V_LOG_INFO << "  per-element          over GTS:                  " << m_loads[0] / m_loads[3];
EDGE_V_LOG_INFO << "  per-element          over grouped:              " << m_loads[1] / m_loads[3];
EDGE_V_LOG_INFO << "  per-element          over grouped (normalized): " << m_loads[2] / m_loads[3];
}

void edge_v::time::Groups::nGroupEls( t_idx   i_first,
t_idx   i_nEls,
t_idx * o_nGroupEls ) const {
nGroupEls( i_first,
i_nEls,
m_nGroups,
m_elTg,
o_nGroupEls );
}