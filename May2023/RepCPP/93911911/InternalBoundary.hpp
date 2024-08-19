
#ifndef EDGE_IO_INTERNAL_BOUNDARY_HPP
#define EDGE_IO_INTERNAL_BOUNDARY_HPP

#include "constants.hpp"
#include "io/logging.h"
#include "data/Dynamic.h"
#include "FileSystem.hpp"
#include "linalg/Mappings.hpp"
#include "sc/SubGrid.hpp"
#include "submodules/visit_writer/visit_writer.h"

namespace edge {
namespace io {
template< typename       TL_T_LID,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class InternalBoundary;
}
}


template< typename       TL_T_LID,
t_entityType   TL_T_EL,
unsigned short TL_O_SP >
class edge::io::InternalBoundary {
private:
static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

static unsigned short const TL_N_FA_VES = C_ENT[TL_T_EL].N_FACE_VERTICES;

static unsigned short const TL_N_EL_VES = C_ENT[TL_T_EL].N_VERTICES;

static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

static unsigned short const TL_N_FA_SVS = CE_N_SUB_VERTICES( C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP );

static unsigned short const TL_N_SFS = CE_N_SUB_FACES( TL_T_EL, TL_O_SP );

static unsigned short const TL_N_SCS  = CE_N_SUB_CELLS( TL_T_EL, TL_O_SP );

float (* m_svCrds)[TL_N_FA_SVS][3];

TL_T_LID (* m_bfSfSv)[TL_N_SFS][TL_N_FA_VES];

float *m_buffer;

float **m_bPtrs;

std::string m_outPath;

unsigned int m_writeStep = 0;

bool const m_binary;

int m_visitElType;


template< typename TL_T_REAL >
void copy( TL_T_LID        i_first,
TL_T_LID        i_size,
unsigned short  i_nQts,
unsigned short  i_stride,
TL_T_REAL      *i_data ) {
EDGE_CHECK_GE( i_first,  0 );
EDGE_CHECK_GE( i_size,   0 );
EDGE_CHECK_GT( i_nQts,   0 );
EDGE_CHECK_GE( i_stride, 1 );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_bf = i_first; l_bf < i_first+i_size; l_bf++ ) {
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
for( unsigned short l_qt = 0; l_qt < i_nQts; l_qt++ ) {
std::size_t l_bId  = l_qt * std::size_t(i_size) * TL_N_SFS;    
l_bId += (l_bf - i_first) * std::size_t(TL_N_SFS); 
l_bId += l_sf;                                     

std::size_t l_dId  = l_bf * std::size_t(i_stride) * TL_N_SFS; 
l_dId += l_sf * i_stride;                         
l_dId += l_qt;                                    

m_buffer[l_bId] = i_data[l_dId];
}
}
}

for( unsigned short l_qt = 0; l_qt < i_nQts; l_qt++ )
m_bPtrs[l_qt] = m_buffer + l_qt * std::size_t(i_size) * TL_N_SFS;
}

public:

InternalBoundary( std::string &i_outPath,
bool i_binary=1 ): m_binary(i_binary) {
if( i_outPath != "") {
std::string l_dir, l_file;
FileSystem::splitPathLast( i_outPath, l_dir, l_file );

if( parallel::g_rank == 0 ) {
for( int l_ra = 0; l_ra < parallel::g_nRanks; l_ra++ ) {
std::string l_dirCreate = l_dir + '/' + std::to_string(l_ra);
FileSystem::createDir( l_dirCreate );
}
}
#ifdef PP_USE_MPI
MPI_Barrier( MPI_COMM_WORLD );
#endif

l_dir = l_dir + "/" + std::to_string(parallel::g_rank) + '/';
m_outPath = l_dir + l_file;
}

if(       C_ENT[TL_T_EL].TYPE_FACES == LINE   ) m_visitElType = VISIT_LINE;
else if ( C_ENT[TL_T_EL].TYPE_FACES == TRIA3  ) m_visitElType = VISIT_TRIANGLE;
else if ( C_ENT[TL_T_EL].TYPE_FACES == QUAD4R ) m_visitElType = VISIT_QUAD;
else if ( C_ENT[TL_T_EL].TYPE_FACES == HEX8R  ) m_visitElType = VISIT_HEXAHEDRON;
else if ( C_ENT[TL_T_EL].TYPE_FACES == TET4   ) m_visitElType = VISIT_TETRA;
else EDGE_LOG_FATAL << "missing element type " << C_ENT[TL_T_EL].TYPE_FACES;
}


void alloc( TL_T_LID             i_nBf,
unsigned short       i_nMaxQts,
edge::data::Dynamic &io_dynMem ) {
if( i_nBf     == 0 ) return;
if( i_nMaxQts == 0 ) return;

std::size_t l_veCrdsSize  = i_nBf * std::size_t(TL_N_FA_SVS);
l_veCrdsSize *= std::size_t(3) * sizeof( float );
m_svCrds = (float (*)[TL_N_FA_SVS][3]) io_dynMem.allocate( l_veCrdsSize );

std::size_t l_faSfSvSize  = i_nBf * std::size_t(TL_N_SFS);
l_faSfSvSize *= std::size_t( TL_N_FA_VES ) * sizeof( TL_T_LID );
m_bfSfSv = (TL_T_LID (*)[TL_N_SFS][TL_N_FA_VES]) io_dynMem.allocate( l_faSfSvSize );

std::size_t l_bufferSize  = i_nBf * std::size_t(TL_N_SFS);
l_bufferSize *= std::size_t(i_nMaxQts) * sizeof( TL_T_LID );
m_buffer = (float*) io_dynMem.allocate( l_bufferSize );

std::size_t l_bPtrs  = i_nMaxQts;
l_bPtrs *= sizeof(float*);
m_bPtrs = (float**) io_dynMem.allocate( l_bPtrs );
}


template< typename TL_T_CHARS_SV,
typename TL_T_CHARS_VE,
typename TL_T_CHARS_BF >
void init( TL_T_LID                    i_nBf,
unsigned short      const   i_scSv[ TL_N_SCS + TL_N_FAS * TL_N_SFS ][ TL_N_EL_VES ],
TL_T_LID            const (*i_bfBe)[2],
TL_T_LID            const  *i_beEl,
TL_T_LID            const (*i_elVe)[TL_N_EL_VES],
TL_T_CHARS_SV       const  *i_charsSv,
TL_T_CHARS_VE       const  *i_charsVe,
TL_T_CHARS_BF       const  *i_charsBf ) {
unsigned short l_faSfSvL[TL_N_FAS][TL_N_SFS][TL_N_FA_VES];
unsigned short l_faSvR[TL_N_FAS][TL_N_FA_SVS];
sc::SubGrid< TL_T_EL,
TL_O_SP >::faSg( i_scSv,
l_faSfSvL,
l_faSvR );

#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( TL_T_LID l_bf = 0; l_bf < i_nBf; l_bf++ ) {
TL_T_LID l_el = i_beEl[ i_bfBe[l_bf][0] ];

unsigned short l_fId = i_charsBf[l_bf].fIdBfEl[0];

float l_veCrds[TL_N_DIS][TL_N_EL_VES];

for( unsigned short l_ve = 0; l_ve < TL_N_EL_VES; l_ve++ ) {
TL_T_LID l_veId = i_elVe[l_el][l_ve];

for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ )
l_veCrds[l_di][l_ve] = i_charsVe[l_veId].coords[l_di];
}

float l_sgCrds[TL_N_FA_SVS][TL_N_DIS];

for( unsigned short l_sv = 0; l_sv < TL_N_FA_SVS; l_sv++ )
for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ )
l_sgCrds[l_sv][l_di] = i_charsSv[ l_faSvR[l_fId][l_sv] ].coords[l_di];

for( unsigned short l_sv = 0; l_sv < TL_N_FA_SVS; l_sv++ ) {
for( unsigned short l_di = 0; l_di < 3; l_di++ )
m_svCrds[l_bf][l_sv][l_di] = 0;

edge::linalg::Mappings::refToPhy( TL_T_EL,
l_veCrds[0],
l_sgCrds[l_sv],
m_svCrds[l_bf][l_sv] );
}
for( unsigned short l_sf = 0; l_sf < TL_N_SFS; l_sf++ ) {
for( unsigned short l_ve = 0; l_ve < TL_N_FA_VES; l_ve++ ) {
m_bfSfSv[l_bf][l_sf][l_ve]  = l_bf * TL_N_FA_SVS;
m_bfSfSv[l_bf][l_sf][l_ve] += l_faSfSvL[l_fId][l_sf][l_ve];
}
}
}
}


template< typename TL_T_REAL >
void write( TL_T_LID         i_first,
TL_T_LID         i_size,
unsigned short   i_nQts,
unsigned short   i_stride,
char const     **i_namesQts,
TL_T_REAL*       i_data ) {
if( i_size == 0 ) return;

EDGE_CHECK_NE( i_stride, 0 );
EDGE_CHECK_GE( i_stride, i_nQts );

copy( i_first,
i_size,
i_nQts,
i_stride,
i_data );

std::string l_file = m_outPath;
l_file += "_" + parallel::g_rankStr;
l_file += "_" + std::to_string((unsigned long long) m_writeStep) + ".vtk";

int l_nPts = i_size * TL_N_FA_SVS;
int l_nCells = i_size * TL_N_SFS;

edge_write_unstructured_mesh( l_file.c_str(),
m_binary,
l_nPts,
m_svCrds[i_first][0],
i_nQts,
l_nCells,
m_visitElType,
m_bfSfSv[i_first][0],
i_namesQts,
m_bPtrs );

m_writeStep++;
}
};

#endif
