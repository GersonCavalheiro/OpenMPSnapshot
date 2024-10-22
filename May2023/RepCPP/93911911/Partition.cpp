
#include "Partition.h"
#ifdef PP_USE_METIS
#include <metis.h>
#endif
#include "../io/logging.h"

edge_v::mesh::Partition::Partition( Mesh           const & i_mesh,
unsigned short const * i_elTg ): m_mesh( i_mesh ),
m_elTg( i_elTg ) {
m_elPa = new t_idx[ m_mesh.nEls() ];
m_elPr = new t_idx[ m_mesh.nEls() ];

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < m_mesh.nEls(); l_el++ ) {
m_elPa[l_el] = 0;
m_elPr[l_el] = 0;
}

getElPr( m_mesh.getTypeEl(),
m_mesh.nEls(),
m_mesh.getElFaEl(),
m_elPa,
m_elTg,
m_elPr );

m_nPas = 1;
m_nPaEls = new t_idx[ 1 ];
m_nPaEls[0] = m_mesh.nEls();

for( t_idx l_el = 0; l_el < m_mesh.nEls(); l_el++ ) {
m_elPa[l_el] = 0;
m_elPr[l_el] = m_elTg[l_el];
}
}

edge_v::mesh::Partition::~Partition() {
delete[] m_elPa;
delete[] m_elPr;
if( m_nPaEls != nullptr ) delete[] m_nPaEls;
}

void edge_v::mesh::Partition::nPaEls( t_idx         i_nEls,
t_idx const * i_elPa,
t_idx       * o_nPaEls ) {
t_idx l_nPas = 0;
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
t_idx l_pa = i_elPa[l_el];
l_nPas = std::max( l_nPas, l_pa );
}
l_nPas++;
for( t_idx l_pa = 0; l_pa < l_nPas; l_pa++ ) o_nPaEls[l_pa] = 0;

for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
t_idx l_pa = i_elPa[l_el];
o_nPaEls[l_pa]++;
}
}


void edge_v::mesh::Partition::getElPr( edge_v::t_entityType         i_elTy,
t_idx                        i_nEls,
t_idx                const * i_elFaEl,
t_idx                const * i_elPa,
unsigned short       const * i_elTg,
t_idx                      * o_elPr ) {
unsigned short l_nTgs = 0;
if( i_elTg != nullptr ) {
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
l_nTgs = std::max( l_nTgs, i_elTg[l_el] );
}
}
l_nTgs++;

auto l_send = [ i_elTy, i_elFaEl, i_elPa ]( t_idx i_el ) {
unsigned short l_nElFas = CE_N_FAS( i_elTy );
bool l_isSend = false;

t_idx l_elPa = i_elPa[i_el];

for( unsigned short l_fa = 0; l_fa < l_nElFas; l_fa++ ) {
t_idx l_ad = i_elFaEl[ i_el*l_nElFas + l_fa ];
if( l_ad != std::numeric_limits< t_idx >::max() ) {
t_idx l_adPa = i_elPa[l_ad];
if( l_elPa != l_adPa ) l_isSend = true;
}
}
return l_isSend;
};

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
o_elPr[l_el] = i_elPa[l_el]*2*l_nTgs;

if( l_send(l_el) ) o_elPr[l_el] += l_nTgs;

if( i_elTg != nullptr ) {
o_elPr[l_el] += i_elTg[l_el];
}
}
}

void edge_v::mesh::Partition::kWay( t_idx          i_nParts,
unsigned short i_nCuts ) {
EDGE_V_CHECK_GT( i_nParts, 1 );

edge_v::t_entityType l_elTy = m_mesh.getTypeEl();
unsigned short l_nElFas = CE_N_FAS( l_elTy );
t_idx l_nEls = m_mesh.nEls();
t_idx const * l_elFaEl = m_mesh.getElFaEl();

idx_t * l_xadj = new idx_t[ l_nEls+1 ];
idx_t * l_adjncy = new idx_t[ l_nEls*l_nElFas ];
getDualGraph( l_elTy,
l_nEls,
l_elFaEl,
l_xadj,
l_adjncy );

idx_t * l_vwgt = nullptr;
idx_t * l_adjwgt = nullptr;
if( m_elTg != nullptr ) {
l_vwgt = new idx_t[ l_nEls ];
l_adjwgt = new idx_t[ l_xadj[l_nEls] ];

getWeights( l_elTy,
l_nEls,
l_xadj[l_nEls],
l_elFaEl,
m_elTg,
l_vwgt,
l_adjwgt );
}

idx_t l_nvtxs = l_nEls;
idx_t l_ncon = 1;
idx_t l_objVal = 0;
idx_t * l_elPa = new idx_t[ l_nEls ];
idx_t l_nParts = i_nParts;
idx_t l_opts[METIS_NOPTIONS];
int l_err = METIS_SetDefaultOptions(l_opts);
EDGE_V_CHECK_EQ( l_err, METIS_OK );
l_opts[METIS_OPTION_CONTIG] = 1;
l_opts[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
l_opts[METIS_OPTION_NCUTS] = i_nCuts;

l_err = METIS_PartGraphKway( &l_nvtxs,
&l_ncon,
l_xadj,
l_adjncy,
l_vwgt,
NULL,
l_adjwgt,
&l_nParts,
NULL,
NULL,
l_opts,
&l_objVal,
l_elPa );
EDGE_V_CHECK_EQ( l_err, METIS_OK );

if( m_elTg != nullptr ) {
delete[] l_vwgt;
delete[] l_adjwgt;
}
delete[] l_adjncy;
delete[] l_xadj;

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < l_nEls; l_el++ ) {
m_elPa[l_el] = l_elPa[l_el];
}

delete[] l_elPa;

getElPr( m_mesh.getTypeEl(),
m_mesh.nEls(),
m_mesh.getElFaEl(),
m_elPa,
m_elTg,
m_elPr );

m_nPas = i_nParts;
if( m_nPaEls != nullptr ) delete[] m_nPaEls;
m_nPaEls = new t_idx[ i_nParts ];
nPaEls( m_mesh.nEls(),
m_elPa,
m_nPaEls );
}