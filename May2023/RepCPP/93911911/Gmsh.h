
#ifndef EDGE_V_IO_GMSH_H
#define EDGE_V_IO_GMSH_H

#include "../constants.h"
#include <string>
#include <vector>

namespace edge_v {
namespace io {
class Gmsh;
}
}


class edge_v::io::Gmsh {
private:
std::vector< double > m_veCrds;

std::vector< std::size_t > m_veTags;

std::vector< std::size_t > m_elTags;

std::vector< std::size_t > m_elVeTags;

std::vector< std::size_t > m_elFaVeTags;

std::vector< int > m_physicalGroupsFa;

std::vector< std::vector< std::size_t > > m_faVeTagsPhysical;


static int getGmshType( t_entityType i_enTy );


static t_entityType getEntityType( int i_gmshType );


static t_idx getId( std::size_t                        i_value,
std::vector< std::size_t > const & i_sortedValues );

public:

Gmsh();


~Gmsh();


void setNumber( std::string const & i_name,
double              i_value ) const;


void open( std::string const & i_pathToFile );


void readMesh();


void write( std::string const & i_pathToFile );


t_entityType getElType() const;


t_idx nVes() const;


t_idx nEls() const;


t_idx nPhysicalGroupsFa() const;


int const * getPhysicalGroupsFa() const;


t_idx nFas( int i_physicalGroupFa ) const;


void getVeCrds( double (*o_veCrds)[3] ) const;


void getFaVe( int     i_physicalGroupFa,
t_idx * o_faVe ) const;


void getElVe( t_idx * o_elVe ) const;


void getElFaVe( t_idx * o_elFaVe ) const;


void reorder( t_idx const * i_priorities );


void partition( t_idx         i_nPas,
t_idx const * i_nPaEls ) const;


void writeElData( std::string           const & i_name,
std::vector< double > const & i_elData,
std::string           const & i_pathToFile  ) const;


template< typename T >
void writeElData( std::string const & i_name,
T           const * i_elData,
std::string const & i_pathToFile  ) const {
std::vector< double > l_elData;
l_elData.resize( m_elTags.size() );
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_el = 0; l_el < m_elTags.size(); l_el++ ) {
l_elData[l_el] = i_elData[l_el];
}

writeElData( i_name,
l_elData,
i_pathToFile );
}
};

#endif