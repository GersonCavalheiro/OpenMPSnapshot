
#ifndef EDGE_PARALLEL_SHARED_H
#define EDGE_PARALLEL_SHARED_H

#include <cstdint>
#include <vector>
#include "data/SparseEntities.hpp"
#include "data/EntityLayout.type"
#include "parallel/global.h"
#include "LoadBalancing.h"
#include "io/logging.h"

namespace edge {
namespace parallel {
class Shared;
}
}

class edge::parallel::Shared {
public:
int m_nWrks;
int m_wrkOff;

typedef enum {
RDY, 
IPR, 
FIN, 
WAI  
} t_status;

private:
struct WrkPkg {
t_status status;

uint64_t padding[8];
};

struct WrkRgn {
unsigned int id;

unsigned short step;

int_tg tg;

int prio;

std::vector< int_spType > spTypes;

std::vector< WrkPkg > wrkPkgs;
};

std::vector< WrkRgn > m_wrkRgns;

LoadBalancing m_balancing;


std::size_t getWrkRgn( unsigned int i_id );


int worker( int i_thread ) {
return i_thread += m_wrkOff;
}

public:

void print();


Shared(){};


~Shared(){};


void init( bool i_separateWrks = true );


bool isWrk();


bool isSched();


bool isComm();


template <typename T = t_vertexChars>
void regWrkRgn( unsigned short   i_tg,
unsigned short   i_step,
unsigned int     i_id,
std::size_t      i_first,
std::size_t      i_size,
int              i_prio=0,
unsigned short   i_nSpTypes=0,
int_spType     * i_spType = nullptr,
const T        * i_enChars = nullptr ) {
#ifdef PP_USE_OMP
#pragma omp barrier
#endif
if( g_thread == 0 ) {
WrkRgn l_wrkRgn;
l_wrkRgn.wrkPkgs.resize( m_nWrks );
l_wrkRgn.tg   = i_tg;
l_wrkRgn.step = i_step;
l_wrkRgn.id   = i_id;
l_wrkRgn.prio = i_prio;

for( int l_td = 0; l_td < m_nWrks; l_td++ ) {
l_wrkRgn.wrkPkgs[l_td].status = WAI;
}

std::size_t l_ps = m_wrkRgns.size();

for( std::size_t l_rg = 0; l_rg < m_wrkRgns.size(); l_rg++ ) {
if( m_wrkRgns[l_rg].prio < l_wrkRgn.prio ) {
l_ps = l_rg;
break;
}
}

m_wrkRgns.insert( m_wrkRgns.begin()+l_ps, l_wrkRgn );

m_balancing.regWrkRgn( l_ps,
i_first,
i_size,
i_nSpTypes,
i_spType,
i_enChars );
}

#ifdef PP_USE_OMP
#pragma omp barrier
#endif
}


bool getWrkTd( unsigned short & o_tg,
unsigned short & o_step,
unsigned int   & o_id,
int_el         & o_first,
int_el         & o_size,
int_el         * o_spEn );


void setStatusTd( t_status     i_status,
unsigned int i_id );


bool getStatusAll( t_status     i_status,
unsigned int i_id );


void setStatusAll( t_status     i_status,
unsigned int i_id );


void resetStatus( t_status i_status );


void balance();


template< typename TL_T_EN >
void numaInit( std::size_t const   i_nEns,
TL_T_EN           * o_arr ) {
int l_worker = worker( g_thread );

if( l_worker < 0 ) {
#ifdef PP_USE_OMP
#pragma omp barrier
#endif

return;
}

std::size_t l_split = i_nEns / std::size_t(m_nWrks);
std::size_t l_rem   = i_nEns % std::size_t(m_nWrks);

TL_T_EN * l_arr = o_arr;
l_arr += l_split * (std::size_t) l_worker;
if( l_rem > 0 ) l_arr += std::min( l_rem, (std::size_t) l_worker );

std::size_t l_nEns = l_split;
if( (std::size_t) l_worker < l_rem ) l_nEns++;

#ifdef PP_USE_OMP
#pragma omp critical
#endif
for( std::size_t l_en = 0; l_en < l_nEns; l_en++ ) l_arr[l_en] = 0;

#ifdef PP_USE_OMP
#pragma omp barrier
#endif
}


template< typename TL_T_LID,
typename TL_T_VA >
void numaInit( unsigned short         i_nTgs,
TL_T_LID       const * i_nTgEnsIn,
TL_T_LID       const * i_nTgEnsSe,
std::size_t    const   i_nVasPerEn,
TL_T_VA              * o_arr ) {
std::size_t l_off = 0;

for( unsigned short l_tg = 0; l_tg < i_nTgs; l_tg++ ) {
std::size_t l_nVas = i_nTgEnsIn[l_tg];
l_nVas *= i_nVasPerEn;

edge::parallel::Shared::numaInit( l_nVas,
o_arr+l_off );
l_off += l_nVas;
}

for( unsigned short l_tg = 0; l_tg < i_nTgs; l_tg++ ) {
std::size_t l_nVas = i_nTgEnsSe[l_tg];
l_nVas *= i_nVasPerEn;

edge::parallel::Shared::numaInit( l_nVas,
o_arr+l_off );
l_off += l_nVas;
}
}
};

#endif
