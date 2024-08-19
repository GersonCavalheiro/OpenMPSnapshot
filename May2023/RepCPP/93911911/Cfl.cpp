
#include "Cfl.h"

#include "../io/logging.h"

edge_v::time::Cfl::Cfl( t_entityType  const    i_elTy,
t_idx                  i_nVes,
t_idx                  i_nEls,
t_idx         const  * i_elVe,
double        const (* i_veCrds)[3],
double        const  * i_inDia,
bool                   i_velModEl,
models::Model        & io_velMod ) {
m_nEls = i_nEls;

t_idx l_size = m_nEls;
m_ts = new double[l_size];

setTimeSteps( i_elTy,
i_nVes,
i_nEls,
i_elVe,
i_veCrds,
i_inDia,
i_velModEl,
io_velMod,
m_tsAbsMin,
m_ts );
}

edge_v::time::Cfl::~Cfl() {
delete[] m_ts;
}

void edge_v::time::Cfl::printStats() const {
EDGE_V_LOG_INFO << "printing CFL-based time step stats (absolute / relative):";

double l_mean = 0;
double l_max = std::numeric_limits< double >::lowest();

for( t_idx l_el = 0; l_el < m_nEls; l_el++ ) {
l_mean += m_ts[l_el];
l_max = std::max( m_ts[l_el], l_max );
}
l_mean /= m_nEls;

EDGE_V_LOG_INFO << "  min:  " << m_tsAbsMin        << " / 1.0";
EDGE_V_LOG_INFO << "  mean: " << l_mean*m_tsAbsMin << " / " << l_mean;
EDGE_V_LOG_INFO << "  max:  " << l_max*m_tsAbsMin  << " / " << l_max;
}

void edge_v::time::Cfl::setTimeSteps( t_entityType  const    i_elTy,
t_idx                  i_nVes,
t_idx                  i_nEls,
t_idx         const  * i_elVe,
double        const (* i_veCrds)[3],
double        const  * i_inDia,
bool                   i_velModEl,
models::Model        & io_velMod,
double               & o_tsAbsMin,
double               * o_ts ) {
unsigned short l_nElVes = CE_N_VES( i_elTy );

o_tsAbsMin = std::numeric_limits< double >::max();

#ifdef PP_USE_OMP
#pragma omp parallel for reduction(min:o_tsAbsMin)
#endif
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
double l_cMean = 0;
if( i_velModEl ) {
l_cMean = io_velMod.getMaxSpeed( l_el );
}
else {
for( unsigned short l_ve = 0; l_ve < l_nElVes; l_ve++ ) {
t_idx l_veId = i_elVe[ l_el*l_nElVes + l_ve ];

l_cMean += io_velMod.getMaxSpeed( l_veId );
}
l_cMean /= l_nElVes;
}
EDGE_V_CHECK_GT( l_cMean, 0 );

o_ts[l_el] = i_inDia[l_el] / l_cMean;

o_tsAbsMin = std::min( o_tsAbsMin, o_ts[l_el] );
}
EDGE_V_CHECK_GT( o_tsAbsMin, 0 );

double l_minInv = 1 / o_tsAbsMin;

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
o_ts[l_el] *= l_minInv;
}
}