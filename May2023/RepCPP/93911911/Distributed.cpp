
#include "Distributed.h"
#ifdef PP_USE_MPI
#include "mpi_wrapper.inc"
#endif
#include <limits>

void edge::parallel::Distributed::sendCommBuffers2( unsigned short   i_tg,
unsigned short & o_cbL,
unsigned short & o_cbLtR,
unsigned short & o_cbGeR ) const {
o_cbL = m_nSendsSync[i_tg]%2;
o_cbLtR = std::numeric_limits< unsigned short >::max();
if( m_nSendsSync[i_tg]%2 == 1 ) o_cbLtR = (m_nSendsSync[i_tg] / 2)%2;
o_cbGeR = m_nSendsSync[i_tg]%2;
}

void edge::parallel::Distributed::recvCommBuffers2( unsigned short   i_tg,
unsigned short & o_cbLtL,
unsigned short & o_cbGeL ) const {
o_cbLtL = (m_nRecvsSync[i_tg]/2)%2;
o_cbGeL = m_nRecvsSync[i_tg]%2;
}

edge::parallel::Distributed::Distributed( int    i_argc,
char * i_argv[] ) {
g_nRanks = 1;
g_rank = 0;
g_rankStr = std::to_string(0);
#ifdef PP_USE_MPI
int l_initialized = 0;
int l_err = MPI_Initialized( &l_initialized );
EDGE_CHECK_EQ( l_err, MPI_SUCCESS );

if( !l_initialized ) {
if( g_nThreads == 1 ) {
MPI_Init( &i_argc,
&i_argv );
}
else {
int l_tdSu;
MPI_Init_thread( &i_argc,
&i_argv,
MPI_THREAD_FUNNELED,
&l_tdSu );
EDGE_CHECK( l_tdSu >= MPI_THREAD_FUNNELED );
}
}

MPI_Comm_size ( MPI_COMM_WORLD, &g_nRanks );
MPI_Comm_rank( MPI_COMM_WORLD, &g_rank );
MPI_Get_version( m_verStd, m_verStd+1 );
g_rankStr = std::to_string( g_rank );

MPI_Barrier( MPI_COMM_WORLD );
#endif
}

void edge::parallel::Distributed::fin() {
#ifdef PP_USE_MPI
MPI_Barrier( MPI_COMM_WORLD );
MPI_Finalize();
#endif
}

std::string edge::parallel::Distributed::getVerStr() {
return std::to_string( m_verStd[0] ) + "." + std::to_string( m_verStd[1] );
}

bool edge::parallel::Distributed::checkSendTgLt( std::size_t    i_ch,
bool           i_lt,
unsigned short i_tg ) const {
bool l_match = (m_sendMsgs[i_ch].tgL == i_tg);
if( m_sendMsgs[i_ch].tgL < m_sendMsgs[i_ch].tgR ) {
l_match = l_match && i_lt;
}
return l_match;
}

bool edge::parallel::Distributed::checkRecvTgLt( std::size_t    i_ch,
bool           i_lt,
unsigned short i_tg ) const {
bool l_match = (m_recvMsgs[i_ch].tgL == i_tg);
if( m_recvMsgs[i_ch].tgL < m_recvMsgs[i_ch].tgR ) {
l_match = l_match && i_lt;
}
return l_match;
}

void edge::parallel::Distributed::init( unsigned short         i_nTgs,
unsigned short         i_nElFas,
std::size_t            i_nEls,
std::size_t            i_nByFa,
std::size_t    const * i_commStruct,
unsigned short const * i_sendFa,
std::size_t    const * i_sendEl,
unsigned short const * i_recvFa,
std::size_t    const * i_recvEl,
unsigned short         i_nCommBuffers,
data::Dynamic        & io_dynMem ) {
m_nTgs = i_nTgs;
m_nCommBuffers = i_nCommBuffers;
EDGE_CHECK( m_nCommBuffers == 1 || m_nCommBuffers == 2 );

m_nSendsSync = (std::size_t *) io_dynMem.allocate( m_nTgs * sizeof(std::size_t) );
m_nRecvsSync = (std::size_t *) io_dynMem.allocate( m_nTgs * sizeof(std::size_t) );

if( i_commStruct != nullptr ) {
m_nChs = i_commStruct[0];
}
m_nSeRe = (std::size_t *) io_dynMem.allocate( m_nChs * sizeof(std::size_t) );

std::size_t l_sizeSend = 0;
std::size_t l_sizeRecv = 0;
for( std::size_t l_ch = 0; l_ch < m_nChs; l_ch++ ) {
std::size_t l_tg    = i_commStruct[1 + l_ch*4 + 0];
std::size_t l_tgAd  = i_commStruct[1 + l_ch*4 + 2];
std::size_t l_nSeRe = i_commStruct[1 + l_ch*4 + 3];
m_nSeRe[l_ch] = l_nSeRe;

l_sizeSend += (l_tg > l_tgAd) ? 2 * l_nSeRe * i_nByFa : l_nSeRe * i_nByFa;
l_sizeRecv += (l_tg < l_tgAd) ? 2 * l_nSeRe * i_nByFa : l_nSeRe * i_nByFa;
}

l_sizeSend *= sizeof(unsigned char);
l_sizeRecv *= sizeof(unsigned char);
m_sendBufferSize = l_sizeSend;
m_sendBuffers = (unsigned char*) io_dynMem.allocate( l_sizeSend*m_nCommBuffers );
m_recvBufferSize = l_sizeRecv;
m_recvBuffers = (unsigned char*) io_dynMem.allocate( l_sizeRecv*m_nCommBuffers );

unsigned short l_nPtrsSend = m_nCommBuffers;
unsigned short l_nPtrsRecv = (m_nCommBuffers == 2) ? 4 : 1;

std::size_t l_sizePtrs = i_nEls * i_nElFas * sizeof(unsigned char*);
unsigned char** l_sendPtrs = (unsigned char** ) io_dynMem.allocate( l_sizePtrs * l_nPtrsSend );
unsigned char** l_recvPtrs = (unsigned char** ) io_dynMem.allocate( l_sizePtrs * l_nPtrsRecv );

for( unsigned short l_po = 0; l_po < l_nPtrsSend; l_po++ )
m_sendPtrs[l_po] = l_sendPtrs + l_po * i_nEls * i_nElFas;
for( unsigned short l_po = 0; l_po < l_nPtrsRecv; l_po++ )
m_recvPtrs[l_po] = l_recvPtrs + l_po * i_nEls * i_nElFas;

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_el = 0; l_el < i_nEls; l_el++ ) {
for( unsigned short l_fa = 0; l_fa < i_nElFas; l_fa++ ) {
for( unsigned short l_po = 0; l_po < l_nPtrsSend; l_po++ )
m_sendPtrs[l_po][l_el*i_nElFas + l_fa] = nullptr;
for( unsigned short l_po = 0; l_po < l_nPtrsRecv; l_po++ )
m_recvPtrs[l_po][l_el*i_nElFas + l_fa] = nullptr;
}
}

m_sendMsgs = (t_msg*) io_dynMem.allocate( m_nChs * sizeof(t_msg) );
m_recvMsgs = (t_msg*) io_dynMem.allocate( m_nChs * sizeof(t_msg) );

std::size_t l_offSend = 0;
std::size_t l_offRecv = 0;
std::size_t l_first = 0;
for( std::size_t l_ch = 0; l_ch < m_nChs; l_ch++ ) {
std::size_t l_tg    = i_commStruct[1 + l_ch*4 + 0];
std::size_t l_raAd  = i_commStruct[1 + l_ch*4 + 1];
std::size_t l_tgAd  = i_commStruct[1 + l_ch*4 + 2];
std::size_t l_nSeRe = i_commStruct[1 + l_ch*4 + 3];

l_sizeSend = (l_tg > l_tgAd) ? l_nSeRe * i_nByFa * 2 : l_nSeRe * i_nByFa;
l_sizeRecv = (l_tg < l_tgAd) ? l_nSeRe * i_nByFa * 2 : l_nSeRe * i_nByFa;

m_sendMsgs[l_ch].tgL     = l_tg;
m_sendMsgs[l_ch].tgR     = l_tgAd;
m_sendMsgs[l_ch].rank    = l_raAd;
m_sendMsgs[l_ch].tag     = l_tg*i_nTgs + l_tgAd;
m_sendMsgs[l_ch].size    = l_sizeSend;
m_sendMsgs[l_ch].offL    = l_offSend;

m_recvMsgs[l_ch].tgL     = l_tg;
m_recvMsgs[l_ch].tgR     = l_tgAd;
m_recvMsgs[l_ch].rank    = l_raAd;
m_recvMsgs[l_ch].tag     = l_tgAd*i_nTgs + l_tg;
m_recvMsgs[l_ch].size    = l_sizeRecv;
m_recvMsgs[l_ch].offL    = l_offRecv;

for( std::size_t l_co = 0; l_co < l_nSeRe; l_co++ ) {
std::size_t l_seFa = i_sendFa[l_first+l_co];
std::size_t l_seEl = i_sendEl[l_first+l_co];
std::size_t l_reFa = i_recvFa[l_first+l_co];
std::size_t l_reEl = i_recvEl[l_first+l_co];

for( unsigned short l_po = 0; l_po < l_nPtrsSend; l_po++ )
m_sendPtrs[l_po][l_seEl*i_nElFas + l_seFa] = m_sendBuffers + (l_po%2)*m_sendBufferSize + l_offSend;

for( unsigned short l_po = 0; l_po < l_nPtrsRecv; l_po++ ) {
if( l_tg >= l_tgAd ) {
m_recvPtrs[l_po][l_reEl*i_nElFas + l_reFa] = m_recvBuffers + (l_po%2)*m_recvBufferSize + l_offRecv;
}
else {
m_recvPtrs[l_po][l_reEl*i_nElFas + l_reFa] = m_recvBuffers + ( (l_po/2)%2 )*m_recvBufferSize + l_offRecv;
}
}

l_offSend += (l_tg > l_tgAd) ? i_nByFa * 2 : i_nByFa;
l_offRecv += (l_tg < l_tgAd) ? i_nByFa * 2 : i_nByFa;
}
l_first += l_nSeRe;
}

reset();
}

void edge::parallel::Distributed::reset() {
for( unsigned short l_tg = 0; l_tg < m_nTgs; l_tg++ ) {
m_nRecvsSync[l_tg] = std::numeric_limits< std::size_t >::max();
m_nSendsSync[l_tg] = std::numeric_limits< std::size_t >::max();
}
}

void edge::parallel::Distributed::min( std::size_t      i_nVals,
double         * i_vals,
unsigned short * o_min ){
for( std::size_t l_va = 0; l_va < i_nVals; l_va++ )
o_min[l_va] = 1;

#ifdef PP_USE_MPI
std::vector< double > l_gVals;
l_gVals.resize( i_nVals );

MPI_Allreduce( i_vals, &l_gVals[0], i_nVals, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

std::vector< int > l_bidsIn( i_nVals );
std::vector< int > l_bidsOut( i_nVals );

for( std::size_t l_va = 0; l_va < i_nVals; l_va++ ) {
if( i_vals[l_va] == l_gVals[l_va] )
l_bidsIn[l_va] = parallel::g_rank;
else
l_bidsIn[l_va] = std::numeric_limits< int >::max();
}

MPI_Allreduce( &l_bidsIn[0], &l_bidsOut[0], i_nVals, MPI_INT, MPI_MIN, MPI_COMM_WORLD );

for( std::size_t l_va = 0; l_va < i_nVals; l_va++ )
if( l_bidsOut[l_va] != parallel::g_rank ) o_min[l_va] = 0;
#endif
}

void edge::parallel::Distributed::syncData( std::size_t           i_nByCh,
std::size_t           i_nByFa,
unsigned char const * i_sendData,
unsigned char       * o_recvData ) {
#ifdef PP_USE_MPI
MPI_Request * l_sendReqs = new MPI_Request[ m_nChs ];
MPI_Request * l_recvReqs = new MPI_Request[ m_nChs ];

unsigned char const * l_sendPtr = i_sendData;
unsigned char       * l_recvPtr = o_recvData;

for( std::size_t l_ch = 0; l_ch < m_nChs; l_ch++ ) {
std::size_t l_size = m_nSeRe[l_ch] * i_nByFa + i_nByCh;

int l_err = MPI_Irecv( l_recvPtr,
l_size,
MPI_BYTE,
m_recvMsgs[l_ch].rank,
m_recvMsgs[l_ch].tag,
MPI_COMM_WORLD,
l_recvReqs+l_ch );
EDGE_CHECK_EQ( l_err, MPI_SUCCESS );

l_err = MPI_Isend( l_sendPtr,
l_size,
MPI_BYTE,
m_sendMsgs[l_ch].rank,
m_sendMsgs[l_ch].tag,
MPI_COMM_WORLD,
l_sendReqs+l_ch );
EDGE_CHECK_EQ( l_err, MPI_SUCCESS );

l_sendPtr += l_size;
l_recvPtr += l_size;
}

int l_err = MPI_Waitall( m_nChs,
l_recvReqs,
MPI_STATUSES_IGNORE );
EDGE_CHECK_EQ( l_err, MPI_SUCCESS );
l_err = MPI_Waitall( m_nChs,
l_sendReqs,
MPI_STATUSES_IGNORE );
EDGE_CHECK_EQ( l_err, MPI_SUCCESS );

delete[] l_sendReqs;
delete[] l_recvReqs;
#endif
}