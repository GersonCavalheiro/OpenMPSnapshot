
#ifndef EDGE_V_IO_HDF5_H
#define EDGE_V_IO_HDF5_H

#include <hdf5.h>
#include <string>
#include "logging.h"
#include "../constants.h"

namespace edge_v {
namespace io {
class Hdf5;
}
}


class edge_v::io::Hdf5 {
private:
hid_t m_fileId = 0;

std::string m_groupStr;


template< typename TL_T_IN,
typename TL_T_OUT >
static void convert( std::size_t         i_nValues,
TL_T_IN     const * i_data,
TL_T_OUT          * o_data ) {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( std::size_t l_va = 0; l_va < i_nValues; l_va++ ) {
#if !defined(__clang__) && !defined(__INTEL_COMPILER)
if( !std::is_signed<TL_T_OUT>() ) {
EDGE_V_CHECK_GE( i_data[l_va], std::numeric_limits< TL_T_OUT >::lowest() );
}
EDGE_V_CHECK_LE( i_data[l_va], std::numeric_limits< TL_T_OUT >::max()    );
#endif

o_data[l_va] = i_data[l_va];
}
}


void set( std::string const & i_name,
t_idx               i_nValues,
void        const * i_data,
hid_t               i_memType,
hid_t               i_fileType ) const;


void get( std::string const & i_name,
hid_t               i_memType,
void              * o_data ) const;

public:

Hdf5( std::string const & i_path,
bool                i_readOnly = true );


~Hdf5();


bool exists( std::string const & i_path ) const;


t_idx nVas( std::string const & i_name ) const;


void createGroup( std::string const & i_group ) const;


void set( std::string    const & i_name,
t_idx                  i_nValues,
unsigned short const * i_data ) const;


void set( std::string const & i_name,
t_idx               i_nValues,
t_idx       const * i_data ) const;


void set( std::string const & i_name,
t_idx               i_nValues,
float       const * i_data ) const;


void set( std::string const & i_name,
t_idx               i_nValues,
double      const * i_data ) const;


void get( std::string    const & i_name,
unsigned short       * o_data ) const;


void get( std::string const & i_name,
t_idx             * o_data ) const;


void get( std::string const & i_name,
float             * o_data ) const;


void get( std::string const & i_name,
double            * o_data ) const;
};

#endif