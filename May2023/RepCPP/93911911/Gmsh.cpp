
#include "Gmsh.h"
#include <gmsh.h>
#include "logging.h"

int edge_v::io::Gmsh::getGmshType( t_entityType i_enTy ) {
int l_ty = std::numeric_limits< int >::max();

if( i_enTy == POINT ) {
l_ty = gmsh::model::mesh::getElementType( "point",
1 );
}
else if( i_enTy == LINE ) {
l_ty = gmsh::model::mesh::getElementType( "line",
1 );
}
else if( i_enTy == QUAD4R ) {
l_ty = gmsh::model::mesh::getElementType( "quadrangle",
1 );
}
else if( i_enTy == TRIA3 ) {
l_ty = gmsh::model::mesh::getElementType( "triangle",
1 );
}
else if( i_enTy == HEX8R ) {
l_ty = gmsh::model::mesh::getElementType( "hexahedron",
1 );
}
else if( i_enTy == TET4 ) {
l_ty = gmsh::model::mesh::getElementType( "tetrahedron",
1 );
}
else EDGE_V_LOG_FATAL;

return l_ty;
}

edge_v::t_entityType edge_v::io::Gmsh::getEntityType( int i_gmshType ) {
t_entityType l_ty = POINT;

if( i_gmshType == getGmshType( LINE ) ) {
l_ty = LINE;
}
else if( i_gmshType == getGmshType( QUAD4R ) ) {
l_ty = QUAD4R;
}
else if( i_gmshType == getGmshType( TRIA3 ) ) {
l_ty = TRIA3;
}
else if( i_gmshType == getGmshType( HEX8R ) ) {
l_ty = HEX8R;
}
else if( i_gmshType == getGmshType( TET4 ) ) {
l_ty = TET4;
}
else {
EDGE_V_CHECK_EQ( i_gmshType, getGmshType( l_ty ) );
}

return l_ty;
}

edge_v::t_idx edge_v::io::Gmsh::getId( std::size_t                        i_value,
std::vector< std::size_t > const & i_sortedValues ) {
auto l_lowerBound = std::lower_bound( i_sortedValues.begin(),
i_sortedValues.end(),
i_value );

EDGE_V_CHECK_NE( l_lowerBound, i_sortedValues.end() );
EDGE_V_CHECK_EQ( *l_lowerBound, i_value );

return l_lowerBound - i_sortedValues.begin();
}

void edge_v::io::Gmsh::readMesh() {
m_veCrds.resize(0);
m_veTags.resize(0);
m_elTags.resize(0);
m_elVeTags.resize(0);
m_elFaVeTags.resize(0);
m_physicalGroupsFa.resize(0);
m_faVeTagsPhysical.resize(0);

t_entityType   l_elTy   = getElType();
t_entityType   l_faTy   = CE_T_FA(  l_elTy );
unsigned short l_nFaVes = CE_N_VES( l_faTy );
unsigned short l_nElFas = CE_N_FAS( l_elTy );
unsigned short l_nElVes = CE_N_VES( l_elTy );
unsigned short l_nDis   = CE_N_DIS( l_elTy );

gmsh::model::mesh::renumberElements();

std::vector< double > l_parametricCoord;
std::vector< std::size_t > l_veTags;
std::vector< double > l_veCrds;
gmsh::model::mesh::getNodes( l_veTags,
l_veCrds,
l_parametricCoord,
-1,
-1,
true,
false );
l_parametricCoord.resize(0);

std::vector< std::size_t > l_vePerm( l_veTags.size() );
for( std::size_t l_ve = 0; l_ve < l_vePerm.size(); l_ve++ ) {
l_vePerm[l_ve] = l_ve;
}
std::sort( l_vePerm.begin(),
l_vePerm.end(),
[&]( std::size_t const & i_v0, std::size_t const & i_v1 )  {
return (l_veTags[i_v0] < l_veTags[i_v1]); });

m_veTags.reserve( l_veTags.size() );
m_veCrds.reserve( l_veCrds.size() );
for( std::size_t l_ve = 0; l_ve < l_vePerm.size(); l_ve++ ) {
std::size_t l_pe = l_vePerm[l_ve];
std::size_t l_tag = l_veTags[l_pe];

if( m_veTags.size() != 0 && m_veTags.back() == l_tag ) continue;

m_veTags.push_back( l_tag );
for( unsigned short l_di = 0; l_di < 3; l_di++ ) {
m_veCrds.push_back( l_veCrds[l_pe*3 + l_di] );
}
}
l_veTags.resize(0);
l_vePerm.resize(0);

EDGE_V_CHECK_EQ( m_veTags.size()*3, m_veCrds.size() );
for( std::size_t l_ve = 1; l_ve < m_veTags.size(); l_ve++ ) {
EDGE_V_CHECK_LT( m_veTags[l_ve-1], m_veTags[l_ve] ) << m_veTags[l_ve-1] << " " << m_veTags[l_ve];
}

std::vector< int > l_elTys;
std::vector< std::vector< std::size_t > > l_elTags;
std::vector< std::vector< std::size_t > > l_elVeTags;
gmsh::model::mesh::getElements( l_elTys,
l_elTags,
l_elVeTags,
l_nDis );

EDGE_V_CHECK_EQ( l_elTys.size(), 1 );
EDGE_V_CHECK_EQ( l_elTags.size(), 1 );
EDGE_V_CHECK_EQ( l_elVeTags.size(), 1 );
EDGE_V_CHECK_EQ( l_elTys[0], getGmshType( l_elTy ) );
EDGE_V_CHECK_EQ( l_elVeTags[0].size(), l_elTags[0].size()*l_nElVes );

m_elTags.insert( m_elTags.end(),
l_elTags[0].begin(),
l_elTags[0].end() );

m_elVeTags.insert( m_elVeTags.end(),
l_elVeTags[0].begin(),
l_elVeTags[0].end() );

for( std::size_t l_el = 1; l_el < m_elTags.size(); l_el++ ) {
EDGE_V_CHECK_LT( m_elTags[l_el-1], m_elTags[l_el] );
}

if( l_nDis == 2 ) {
gmsh::model::mesh::getElementEdgeNodes( getGmshType(l_elTy),
m_elFaVeTags );
}
else {
int l_gmshFaTy = (l_faTy == TRIA3) ? 3 : 4;

gmsh::model::mesh::getElementFaceNodes( getGmshType(l_elTy),
l_gmshFaTy,
m_elFaVeTags );
}
EDGE_V_CHECK_EQ( m_elFaVeTags.size(), m_elTags.size()*l_nElFas*l_nFaVes );

gmsh::vectorpair l_physicalGroups;
gmsh::model::getPhysicalGroups( l_physicalGroups );

for( std::size_t l_pg = 0; l_pg < l_physicalGroups.size(); l_pg++ ) {
if( l_physicalGroups[l_pg].first == l_nDis-1 ) {
m_physicalGroupsFa.push_back( l_physicalGroups[l_pg].second );
}
}

m_faVeTagsPhysical.resize( m_physicalGroupsFa.size() );

for( std::size_t l_pg = 0; l_pg < m_physicalGroupsFa.size(); l_pg++ ) {
std::vector< int > l_entities;
gmsh::model::getEntitiesForPhysicalGroup( l_nDis-1,
m_physicalGroupsFa[l_pg],
l_entities );

for( std::size_t l_en = 0; l_en < l_entities.size(); l_en++ ) {
std::vector< int > l_faTys;
std::vector< std::vector< std::size_t > > l_faTags;
std::vector< std::vector< std::size_t > > l_faVeTags;

gmsh::model::mesh::getElements( l_faTys,
l_faTags,
l_faVeTags,
l_nDis-1,
l_entities[l_en] );

if( l_faTys.size() == 0 ) {
EDGE_V_CHECK_EQ( l_faVeTags.size(), 0 );
continue;
}

EDGE_V_CHECK_EQ( l_faTys.size(), 1 );
EDGE_V_CHECK_EQ( l_faTys[0], getGmshType(l_faTy) );
EDGE_V_CHECK_GT( l_faVeTags.size(), 0 );

m_faVeTagsPhysical[l_pg].reserve( m_faVeTagsPhysical[l_pg].size() + l_faVeTags[0].size() );
m_faVeTagsPhysical[l_pg].insert( m_faVeTagsPhysical[l_pg].end(),
l_faVeTags[0].begin(),
l_faVeTags[0].end() );
}
}
}

edge_v::io::Gmsh::Gmsh() {
gmsh::initialize();
setNumber( "General.Verbosity",
0 );
}

edge_v::io::Gmsh::~Gmsh() {
gmsh::finalize();
}

void edge_v::io::Gmsh::setNumber( std::string const & i_name,
double              i_value ) const {
gmsh::option::setNumber( i_name,
i_value );
}

void edge_v::io::Gmsh::open( std::string const & i_pathToFile ) {
gmsh::open( i_pathToFile );
}

void edge_v::io::Gmsh::write( std::string const & i_pathToFile ) {
gmsh::write( i_pathToFile );
}


edge_v::t_entityType edge_v::io::Gmsh::getElType() const {
int l_nDis = gmsh::model::getDimension();

std::vector< int > l_types;
gmsh::model::mesh::getElementTypes( l_types, l_nDis );
EDGE_V_CHECK_EQ( l_types.size(), 1 );

return getEntityType( l_types[0] );
}

edge_v::t_idx edge_v::io::Gmsh::nVes() const {
return m_veTags.size();
}

edge_v::t_idx edge_v::io::Gmsh::nEls() const {
return m_elTags.size();
}

void edge_v::io::Gmsh::getVeCrds( double (*o_veCrds)[3] ) const {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_ve = 0; l_ve < m_veTags.size(); l_ve++ ) {
for( unsigned short l_di = 0; l_di < 3; l_di++ ) {
o_veCrds[l_ve][l_di] = m_veCrds[l_ve*3 + l_di];
}
}
}

void edge_v::io::Gmsh::getElVe( t_idx * o_elVe ) const {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_en = 0; l_en < m_elVeTags.size(); l_en++ ) {
o_elVe[l_en] = getId( m_elVeTags[l_en],
m_veTags );
}
}

void edge_v::io::Gmsh::getElFaVe( t_idx * o_elFaVe ) const {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_en = 0; l_en < m_elFaVeTags.size(); l_en++ ) {
o_elFaVe[l_en] = getId( m_elFaVeTags[l_en],
m_veTags );
}
}

edge_v::t_idx edge_v::io::Gmsh::nPhysicalGroupsFa() const {
return m_physicalGroupsFa.size();
}

int const * edge_v::io::Gmsh::getPhysicalGroupsFa() const {
return m_physicalGroupsFa.data();
}

edge_v::t_idx edge_v::io::Gmsh::nFas( int i_physicalGroupFa ) const {
t_entityType   l_elTy   = getElType();
t_entityType   l_faTy   = CE_T_FA(  l_elTy );
unsigned short l_nFaVes = CE_N_VES( l_faTy );

std::size_t l_nFas = 0;

for( std::size_t l_pg = 0; l_pg < m_physicalGroupsFa.size(); l_pg++ ) {
if( m_physicalGroupsFa[l_pg] == i_physicalGroupFa ) {
l_nFas = m_faVeTagsPhysical[l_pg].size();
EDGE_V_CHECK_EQ( l_nFas % l_nFaVes, 0 );
l_nFas /= l_nFaVes;
break;
}
}

return l_nFas;
}

void edge_v::io::Gmsh::getFaVe( int     i_physicalGroupFa,
t_idx * o_faVe ) const {
for( std::size_t l_pg = 0; l_pg < m_physicalGroupsFa.size(); l_pg++ ) {
if( m_physicalGroupsFa[l_pg] == i_physicalGroupFa ) {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_en = 0; l_en <  m_faVeTagsPhysical[l_pg].size(); l_en++ ) {
o_faVe[l_en] = getId( m_faVeTagsPhysical[l_pg][l_en],
m_veTags );
}
}
}
}

void edge_v::io::Gmsh::reorder( t_idx const * i_priorities ) {
int l_nDis = gmsh::model::getDimension();
gmsh::vectorpair l_dimTags;
gmsh::model::getEntities( l_dimTags,
l_nDis );
EDGE_V_CHECK_EQ( l_dimTags.size(), 1 )
<< "error, can't reorder: the elements are split into multiple tags; "
<< "try using a single tag, i.e., volume in 3D, for all elements when meshing";

std::vector< std::size_t > l_elPerm;
l_elPerm.resize( m_elTags.size() );
for( std::size_t l_el = 0; l_el < l_elPerm.size(); l_el++ ) {
l_elPerm[l_el] = l_el;
}
std::sort( l_elPerm.begin(),
l_elPerm.end(),
[&]( std::size_t const & i_e0, std::size_t const & i_e1 )  {
return (i_priorities[i_e0] < i_priorities[i_e1]); });

t_entityType l_elTy = getElType();
int l_elTyGmsh = getGmshType( l_elTy );

gmsh::model::mesh::reorderElements( l_elTyGmsh,
-1,
l_elPerm );
}

void edge_v::io::Gmsh::partition( t_idx         i_nPas,
t_idx const * i_nPaEls ) const {
int l_view = gmsh::view::add( "edge_v_partitions" );
std::string l_pluginName = "EdgePartition";
std::string l_pluginOption = "nPaEls";
gmsh::plugin::setNumber( l_pluginName,
l_pluginOption,
l_view+0.5 );

std::vector< double > l_nPaEls;
for( t_idx l_pa = 0; l_pa < i_nPas; l_pa++ ) {
l_nPaEls.push_back( i_nPaEls[l_pa] + 0.5 );
EDGE_V_CHECK_EQ( (t_idx) l_nPaEls.back(), i_nPaEls[l_pa] );
}
gmsh::view::addListData( l_view,
"SP",
i_nPas,
l_nPaEls );

gmsh::plugin::run("EdgePartition");

gmsh::view::remove( l_view );
}

void edge_v::io::Gmsh::writeElData( std::string           const & i_name,
std::vector< double > const & i_elData,
std::string           const & i_pathToFile ) const {
int l_view = gmsh::view::add( i_name );
std::string l_model;
gmsh::model::getCurrent( l_model );

gmsh::view::addHomogeneousModelData( l_view,
0,
l_model,
"ElementData",
m_elTags,
i_elData );

gmsh::view::write( l_view,
i_pathToFile );

gmsh::view::remove( l_view );
}