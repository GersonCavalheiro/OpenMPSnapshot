
#include "Shared.h"
#include "io/logging.h"

#ifdef PP_USE_OMP
#include <omp.h>
#endif

void edge::parallel::Shared::init( bool i_separateWrks ) {
#ifdef PP_USE_OMP
#pragma omp parallel
{
g_thread = omp_get_thread_num();
std::string l_threadStr = std::to_string(g_thread);
l_threadStr.copy( g_threadStr, 10 );

if( g_thread == 0 ) {
g_nThreads = omp_get_num_threads();

if( i_separateWrks == true && g_nThreads > 1 ) {
m_nWrks = g_nThreads - 1;
}
else {
m_nWrks = g_nThreads;
};

EDGE_CHECK( m_nWrks > 0 );
}

#pragma omp barrier

if( m_nWrks == g_nThreads ) {
m_wrkOff = 0;
}
else {
m_wrkOff = -1;
}

EDGE_CHECK( g_thread == omp_get_thread_num() );
}
#else
g_thread   = 0;
g_nThreads = 1;
m_wrkOff  = 0;
m_nWrks    = 1;
#endif

m_balancing.init( m_nWrks );
}

void edge::parallel::Shared::print() {
EDGE_LOG_INFO << "shared memory setup:";
#ifdef PP_USE_OMP
EDGE_LOG_INFO << "  omp-version: "
#if _OPENMP==200505
<< "2.5";
#elif _OPENMP==200805
<< "3.0";
#elif _OPENMP==201107
<< "3.1";
#elif _OPENMP==201307
<< "4.0";
#elif _OPENMP==201511
<< "4.5";
#else
<< "unknown, raw is " << _OPENMP;
#endif

#endif
EDGE_LOG_INFO << "  #threads: " << g_nThreads;
EDGE_LOG_INFO << "  #workers: " << m_nWrks;
}

bool edge::parallel::Shared::isWrk() {
return worker(g_thread) >= 0;
}

bool edge::parallel::Shared::isSched() {
return g_thread == 0;
}

bool edge::parallel::Shared::isComm() {
return g_thread == 0;
}

std::size_t edge::parallel::Shared::getWrkRgn( unsigned int i_id ) {
std::size_t l_rg;
for( l_rg = 0; l_rg < m_wrkRgns.size(); l_rg++ ) {
if( m_wrkRgns[l_rg].id == i_id ) break;
EDGE_CHECK( l_rg != m_wrkRgns.size()-1 );
}
return l_rg;
}

bool edge::parallel::Shared::getWrkTd( unsigned short & o_tg,
unsigned short & o_step,
unsigned int   & o_id,
int_el         & o_first,
int_el         & o_size,
int_el         * o_firstSp ) {
int l_worker = worker(g_thread);

EDGE_CHECK( l_worker >= 0 );

o_tg    = std::numeric_limits< int_tg >::max();
o_step  = std::numeric_limits< unsigned short >::max();
o_id    = std::numeric_limits< unsigned int >::max();
o_first = std::numeric_limits< int_el       >::max();
o_size  = std::numeric_limits< int_el       >::max();

for( std::size_t l_rg = 0; l_rg < m_wrkRgns.size(); l_rg++ ) {
volatile WrkPkg* l_wps = m_wrkRgns[l_rg].wrkPkgs.data();

if( l_wps[l_worker].status == RDY ) {
o_tg    = m_wrkRgns[l_rg].tg;
o_step  = m_wrkRgns[l_rg].step;
o_id    = m_wrkRgns[l_rg].id;

m_balancing.getWrkTd( l_rg,
l_worker,
o_first,
o_size,
o_firstSp );

#ifdef PP_USE_OMP
#pragma omp flush
#endif

return true;
}
}

return false;
}

void edge::parallel::Shared::setStatusAll( t_status     i_status,
unsigned int i_id ) {
#ifdef PP_USE_OMP
#pragma omp flush
#endif

std::size_t l_rg = getWrkRgn( i_id );

volatile WrkPkg* l_wps = m_wrkRgns[l_rg].wrkPkgs.data();

for( int l_wo = 0; l_wo < m_nWrks; l_wo++ ) {
if( i_status == RDY ) {
EDGE_CHECK( l_wps[l_wo].status == FIN || l_wps[l_wo].status == WAI );
}
else {
EDGE_LOG_FATAL << "status change not allowed";
}

l_wps[l_wo].status = i_status;
}
}

void edge::parallel::Shared::resetStatus( t_status i_status ) {
#ifdef PP_USE_OMP
#pragma omp flush
#endif

for( unsigned short l_rg = 0; l_rg < m_wrkRgns.size(); l_rg++ ) {
volatile WrkPkg* l_wps = m_wrkRgns[l_rg].wrkPkgs.data();

for( int l_wo = 0; l_wo < m_nWrks; l_wo++ ) {
l_wps[l_wo].status = i_status;
}
}
}

void edge::parallel::Shared::setStatusTd(  t_status     i_status,
unsigned int i_id ) {
int l_worker = worker( g_thread );

#ifdef PP_USE_OMP
#pragma omp flush
#endif

EDGE_CHECK( l_worker >= 0 );

std::size_t l_rg = getWrkRgn( i_id );

volatile t_status *l_st = &m_wrkRgns[l_rg].wrkPkgs[l_worker].status;

if( i_status == IPR ) {
EDGE_CHECK( *l_st == RDY );
m_balancing.startClock( l_rg, l_worker );
}
else if( i_status == FIN ) {
EDGE_CHECK( *l_st == IPR );
m_balancing.stopClock( l_rg, l_worker );
}
else {
EDGE_LOG_FATAL << "previous status not matching: " << i_status;
}

*l_st = i_status;
}

bool edge::parallel::Shared::getStatusAll( t_status     i_status,
unsigned int i_id ) {
std::size_t l_rg = getWrkRgn( i_id );

volatile WrkPkg* l_wps = m_wrkRgns[l_rg].wrkPkgs.data();

for( int l_wo = 0; l_wo < m_nWrks; l_wo++ ) {
if(l_wps[l_wo].status != i_status) return false;
}

#ifdef PP_USE_OMP
#pragma omp flush
#endif

return true;
}

void edge::parallel::Shared::balance() {
m_balancing.balance();
if( EDGE_VLOG_IS_ON(2) )
m_balancing.print();
}