#pragma once

#include "dg/backend/typedefs.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_collective.h"
#include "mpi_grid.h"
#include "projection.h"



namespace dg
{
namespace detail{
static void global2bufferIdx(
const cusp::array1d<int, cusp::host_memory>& global_idx,
cusp::array1d<int, cusp::host_memory>& buffer_idx,
thrust::host_vector<int>& locally_unique_global_idx)
{
thrust::host_vector<int> index(global_idx.begin(), global_idx.end()), m_global_idx(index);
thrust::sequence( index.begin(), index.end());
thrust::stable_sort_by_key( m_global_idx.begin(), m_global_idx.end(), index.begin());
thrust::host_vector<int> ones( index.size(), 1);
thrust::host_vector<int> unique_global( index.size()), howmany( index.size());
typedef typename thrust::host_vector<int>::iterator iterator;
thrust::pair<iterator, iterator> new_end;
new_end = thrust::reduce_by_key( m_global_idx.begin(), m_global_idx.end(), ones.begin(), unique_global.begin(), howmany.begin());
locally_unique_global_idx.assign( unique_global.begin(), new_end.first);
thrust::host_vector<int> gather_map;
for( int i=0; i<(int)locally_unique_global_idx.size(); i++)
for( int j=0; j<howmany[i]; j++)
gather_map.push_back(i);
assert( gather_map.size() == global_idx.size());
buffer_idx.resize( global_idx.size());
thrust::scatter( gather_map.begin(), gather_map.end(), index.begin(), buffer_idx.begin());
}
}


template<class ConversionPolicy, class real_type>
dg::MIHMatrix_t<real_type> convert( const dg::IHMatrix_t<real_type>& global, const ConversionPolicy& policy)
{
dg::iHVec unique_global_idx;
cusp::array1d<int, cusp::host_memory> buffer_idx;
dg::detail::global2bufferIdx( global.column_indices, buffer_idx, unique_global_idx);
dg::GeneralComm<dg::iHVec, thrust::host_vector<real_type>> comm( unique_global_idx, policy);
if( !comm.isCommunicating() )
{
cusp::array1d<int, cusp::host_memory> local_idx(global.column_indices), pids(local_idx);
bool success = true;
for(unsigned i=0; i<local_idx.size(); i++)
success = policy.global2localIdx(global.column_indices[i], local_idx[i], pids[i]);
assert( success);
dg::IHMatrix_t<real_type> local( global.num_rows, policy.local_size(), global.values.size());
comm = dg::GeneralComm< dg::iHVec, thrust::host_vector<real_type>>();
local.row_offsets=global.row_offsets;
local.column_indices=local_idx;
local.values=global.values;
return dg::MIHMatrix_t<real_type>( local, comm, dg::row_dist);
}
dg::IHMatrix_t<real_type> local( global.num_rows, comm.buffer_size(), global.values.size());
local.row_offsets=global.row_offsets;
local.column_indices=buffer_idx;
local.values=global.values;
dg::MIHMatrix_t<real_type> matrix(   local, comm, dg::row_dist);
return matrix;
}


template<class ConversionPolicy, class real_type>
dg::IHMatrix_t<real_type> convertGlobal2LocalRows( const dg::IHMatrix_t<real_type>& global, const ConversionPolicy& policy)
{
cusp::array1d<int, cusp::host_memory> rows( global.column_indices.size()), local_rows(rows);
for( unsigned i=0; i<global.num_rows; i++)
for( int k = global.row_offsets[i]; k < global.row_offsets[i+1]; k++)
rows[k] = i;
thrust::host_vector<int> pids(rows.size());
bool success = true;
for(unsigned i=0; i<rows.size(); i++)
if( !policy.global2localIdx(rows[i], local_rows[i], pids[i]) ) success = false;
assert( success);
dg::BijectiveComm<dg::iHVec, thrust::host_vector<real_type>> comm( pids, policy.communicator());
auto rowsV = dg::construct<thrust::host_vector<real_type> >( local_rows);
auto colsV = dg::construct<thrust::host_vector<real_type> >( global.column_indices);
auto row_buffer = comm.global_gather( &rowsV[0]);
auto col_buffer = comm.global_gather( &colsV[0]);
auto val_buffer = comm.global_gather( &global.values[0]);
int local_num_rows = dg::blas1::reduce( row_buffer, (real_type)0, thrust::maximum<real_type>())+1;
cusp::coo_matrix<int, real_type, cusp::host_memory> A( local_num_rows, global.num_cols, row_buffer.size());
A.row_indices = dg::construct<cusp::array1d<int, cusp::host_memory>>( row_buffer);
A.column_indices = dg::construct<cusp::array1d<int, cusp::host_memory>>( col_buffer);
A.values = val_buffer;
A.sort_by_row_and_column();
return dg::IHMatrix_t<real_type>(A);
}


template<class ConversionPolicy, class real_type>
void convertLocal2GlobalCols( dg::IHMatrix_t<real_type>& local, const ConversionPolicy& policy)
{
int rank=0;
MPI_Comm_rank( policy.communicator(), &rank);

bool success = true;
for(unsigned i=0; i<local.column_indices.size(); i++)
if( !policy.local2globalIdx(local.column_indices[i], rank, local.column_indices[i]) ) success = false;
assert( success);
local.num_cols = policy.size();
}

namespace create
{


template<class real_type>
dg::MIHMatrix_t<real_type> interpolation( const aRealMPITopology2d<real_type>&
g_new, const aRealMPITopology2d<real_type>& g_old,std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
g_new.local(), g_old.global(), method);
return convert(  mat, g_old);
}
template<class real_type>
dg::MIHMatrix_t<real_type> interpolation( const aRealMPITopology3d<real_type>&
g_new, const aRealMPITopology3d<real_type>& g_old,std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
g_new.local(), g_old.global(), method);
return convert(  mat, g_old);
}
template<class real_type>
dg::MIHMatrix_t<real_type> interpolation( const aRealMPITopology3d<real_type>&
g_new, const aRealMPITopology2d<real_type>& g_old,std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
g_new.local(), g_old.global(), method);
return convert(  mat, g_old);
}


template<class real_type>
dg::MIHMatrix_t<real_type> projection( const aRealMPITopology2d<real_type>&
g_new, const aRealMPITopology2d<real_type>& g_old, std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::projection(
g_new.global(), g_old.local(), method);
convertLocal2GlobalCols( mat, g_old);
auto mat_loc = convertGlobal2LocalRows( mat, g_new);
return convert(  mat_loc, g_old);
}
template<class real_type>
dg::MIHMatrix_t<real_type> projection( const aRealMPITopology3d<real_type>&
g_new, const aRealMPITopology3d<real_type>& g_old, std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::projection(
g_new.global(), g_old.local(), method);
convertLocal2GlobalCols( mat, g_old);
auto mat_loc = convertGlobal2LocalRows( mat, g_new);
return convert(  mat_loc, g_old);
}


template<class real_type>
dg::MIHMatrix_t<real_type> interpolation(
const thrust::host_vector<real_type>& x,
const thrust::host_vector<real_type>& y,
const aRealMPITopology2d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
std::string method = "dg")
{
dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y, g.global(),
bcx, bcy, method);
return convert(  mat, g);
}


template<class real_type>
dg::MIHMatrix_t<real_type> interpolation(
const thrust::host_vector<real_type>& x,
const thrust::host_vector<real_type>& y,
const thrust::host_vector<real_type>& z,
const aRealMPITopology3d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
std::string method = "linear")
{
dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y,z,
g.global(), bcx, bcy, bcz, method);
return convert(  mat, g);
}




}
}
