#pragma once
#include <cassert>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "blas1_dispatch_shared.h"
#include "dg/blas1.h"
#include "memory.h"
#include "mpi_communicator.h"

namespace dg{



template<class Index, class Vector>
struct Collective
{
Collective(){
m_comm = MPI_COMM_NULL;
}

Collective( const thrust::host_vector<int>& sendTo, MPI_Comm comm) {
construct( sendTo, comm);
}

void construct( thrust::host_vector<int> sendTo, MPI_Comm comm){
thrust::host_vector<int> recvFrom(sendTo), accS(sendTo), accR(sendTo);
m_comm=comm;
int rank, size;
MPI_Comm_rank( m_comm, &rank);
MPI_Comm_size( m_comm, &size);
assert( sendTo.size() == (unsigned)size);
thrust::host_vector<unsigned> global( size*size);
MPI_Allgather( sendTo.data(), size, MPI_UNSIGNED,
global.data(), size, MPI_UNSIGNED,
m_comm);
for( unsigned i=0; i<(unsigned)size; i++)
recvFrom[i] = global[i*size+rank];
thrust::exclusive_scan( sendTo.begin(),   sendTo.end(),   accS.begin());
thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());
m_sendTo=sendTo, m_recvFrom=recvFrom, m_accS=accS, m_accR=accR;
}

unsigned size() const {return values_size();}
MPI_Comm comm() const {return m_comm;}

void transpose(){ m_sendTo.swap( m_recvFrom);}
void invert(){ m_sendTo.swap( m_recvFrom);}

void scatter( const Vector& values, Vector& store) const;
void gather( const Vector& store, Vector& values) const;
unsigned store_size() const{
if( m_recvFrom.empty())
return 0;
return thrust::reduce( m_recvFrom.begin(), m_recvFrom.end() );
}
unsigned values_size() const{
if( m_sendTo.empty())
return 0;
return thrust::reduce( m_sendTo.begin(), m_sendTo.end() );
}
MPI_Comm communicator() const{return m_comm;}
private:
unsigned sendTo( unsigned pid) const {return m_sendTo[pid];}
unsigned recvFrom( unsigned pid) const {return m_recvFrom[pid];}
#ifdef _DG_CUDA_UNAWARE_MPI
thrust::host_vector<int> m_sendTo,   m_accS;
thrust::host_vector<int> m_recvFrom, m_accR;
dg::Buffer<thrust::host_vector<get_value_type<Vector> >> m_values, m_store;
#else
thrust::host_vector<int> m_sendTo,   m_accS; 
thrust::host_vector<int> m_recvFrom, m_accR; 
#endif 
MPI_Comm m_comm;
};

template< class Index, class Device>
void Collective<Index, Device>::scatter( const Device& values, Device& store) const
{
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
if( std::is_same< get_execution_policy<Device>, CudaTag>::value ) 
{
cudaError_t code = cudaGetLastError( );
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaDeviceSynchronize(); 
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif 
#ifdef _DG_CUDA_UNAWARE_MPI
m_values.data() = values;
m_store.data().resize( store.size());
MPI_Alltoallv(
thrust::raw_pointer_cast( m_values.data().data()),
thrust::raw_pointer_cast( m_sendTo.data()),
thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(),
thrust::raw_pointer_cast( m_store.data().data()),
thrust::raw_pointer_cast( m_recvFrom.data()),
thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
store = m_store.data();
#else
MPI_Alltoallv(
thrust::raw_pointer_cast( values.data()),
thrust::raw_pointer_cast( m_sendTo.data()),
thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(),
thrust::raw_pointer_cast( store.data()),
thrust::raw_pointer_cast( m_recvFrom.data()),
thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
#endif 
}

template< class Index, class Device>
void Collective<Index, Device>::gather( const Device& gatherFrom, Device& values) const
{
values.resize( values_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
if( std::is_same< get_execution_policy<Device>, CudaTag>::value ) 
{
cudaError_t code = cudaGetLastError( );
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaDeviceSynchronize(); 
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif 
#ifdef _DG_CUDA_UNAWARE_MPI
m_store.data() = gatherFrom;
m_values.data().resize( values.size());
MPI_Alltoallv(
thrust::raw_pointer_cast( m_store.data().data()),
thrust::raw_pointer_cast( m_recvFrom.data()),
thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(),
thrust::raw_pointer_cast( m_values.data().data()),
thrust::raw_pointer_cast( m_sendTo.data()),
thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
values = m_values.data();
#else
MPI_Alltoallv(
thrust::raw_pointer_cast( gatherFrom.data()),
thrust::raw_pointer_cast( m_recvFrom.data()),
thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(),
thrust::raw_pointer_cast( values.data()),
thrust::raw_pointer_cast( m_sendTo.data()),
thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
#endif 
}


template< class Index, class Vector>
struct BijectiveComm : public aCommunicator<Vector>
{
BijectiveComm( ) = default;

BijectiveComm( const thrust::host_vector<int>& pids, MPI_Comm comm) {
construct( pids, comm);
}
BijectiveComm( unsigned local_size, thrust::host_vector<int> localIndexMap, thrust::host_vector<int> pidIndexMap, MPI_Comm comm)
{
construct( pidIndexMap, comm);
m_p.transpose();
}
template<class ConversionPolicy>
BijectiveComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
{
thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
bool success = true;
for(unsigned i=0; i<local.size(); i++)
if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;
assert( success);
construct( pids, p.communicator());
m_p.transpose();
}

template<class OtherIndex, class OtherVector>
BijectiveComm( const BijectiveComm<OtherIndex, OtherVector>& src) {
construct( src.get_pids(), src.communicator());
}


const thrust::host_vector<int>& get_pids()const{return m_pids;}
virtual BijectiveComm* clone() const override final {return new BijectiveComm(*this);}
private:
void compute_global_comm(){
if( m_p.communicator()  == MPI_COMM_NULL){
m_global_comm = false;
return;
}
int rank;
MPI_Comm_rank( m_p.communicator(), &rank);
bool local_communicating = false, global_communicating=false;
for( unsigned i=0; i<m_pids.size(); i++)
if( m_pids[i] != rank)
local_communicating = true;
MPI_Allreduce( &local_communicating, &global_communicating, 1,
MPI_C_BOOL, MPI_LOR, m_p.communicator());
m_global_comm = global_communicating;
}
virtual bool do_isCommunicating() const override final{ return m_global_comm;}
virtual MPI_Comm do_communicator() const override final {return m_p.communicator();}
virtual unsigned do_size() const override final { return m_p.store_size();}
virtual Vector do_make_buffer()const override final{
Vector tmp( do_size() );
return tmp;
}
void construct( thrust::host_vector<int> pids, MPI_Comm comm)
{
this->set_local_size( pids.size());
m_pids = pids;
dg::assign( pids, m_idx);
int size;
MPI_Comm_size( comm, &size);
for( unsigned i=0; i<pids.size(); i++)
assert( 0 <= pids[i] && pids[i] < size);
thrust::host_vector<int> index(pids);
thrust::sequence( index.begin(), index.end());
thrust::stable_sort_by_key( pids.begin(), pids.end(), index.begin());
m_idx=index;

thrust::host_vector<int> one( pids.size(), 1), keys(one), number(one);
typedef thrust::host_vector<int>::iterator iterator;
thrust::pair< iterator, iterator> new_end =
thrust::reduce_by_key( pids.begin(), pids.end(), 
one.begin(), keys.begin(), number.begin() );
unsigned distance = thrust::distance( keys.begin(), new_end.first);
thrust::host_vector<int> sendTo( size, 0 );
for( unsigned i=0; i<distance; i++)
sendTo[keys[i]] = number[i];
m_p.construct( sendTo, comm);
m_values.data().resize( m_idx.size());
compute_global_comm();
}
virtual void do_global_gather( const get_value_type<Vector>* values, Vector& store)const override final
{
typename Vector::const_pointer values_ptr(values);
if( m_global_comm)
{
thrust::gather( m_idx.begin(), m_idx.end(), values_ptr, m_values.data().begin());
m_p.scatter( m_values.data(), store);
}
else
thrust::gather( m_idx.begin(), m_idx.end(), values_ptr, store.begin());
}

virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values) const override final
{
typename Vector::pointer values_ptr(values);
if( m_global_comm)
{
m_p.gather( toScatter, m_values.data());
thrust::scatter( m_values.data().begin(), m_values.data().end(), m_idx.begin(), values_ptr);
}
else
{
thrust::scatter( toScatter.begin(), toScatter.end(), m_idx.begin(), values_ptr);
}
}
Buffer<Vector> m_values;
Index m_idx;
Collective<Index, Vector> m_p;
thrust::host_vector<int> m_pids;
bool m_global_comm = false;
};


template< class Index, class Vector>
struct SurjectiveComm : public aCommunicator<Vector>
{
SurjectiveComm(){
m_buffer_size = m_store_size = 0;
}
SurjectiveComm( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm)
{
construct( local_size, localIndexMap, pidIndexMap, comm);
}

template<class ConversionPolicy>
SurjectiveComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
{
thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
bool success = true;
for(unsigned i=0; i<local.size(); i++)
if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;

assert( success);
construct( p.local_size(), local, pids, p.communicator());
}

template<class OtherIndex, class OtherVector>
SurjectiveComm( const SurjectiveComm<OtherIndex, OtherVector>& src)
{
construct( src.local_size(), src.getLocalIndexMap(), src.getPidIndexMap(), src.communicator());
}

const thrust::host_vector<int>& getLocalIndexMap() const {return m_localIndexMap;}
const thrust::host_vector<int>& getPidIndexMap() const {return m_pidIndexMap;}
const Index& getSortedIndexMap() const {return m_sortedIndexMap;}
virtual SurjectiveComm* clone() const override final {return new SurjectiveComm(*this);}
bool isLocalBijective() const {return !m_reduction;}
private:
virtual bool do_isCommunicating() const override final{
return m_bijectiveComm.isCommunicating();
}
virtual Vector do_make_buffer()const override final{
Vector tmp(do_size());
return tmp;
}
virtual void do_global_gather( const get_value_type<Vector>* values, Vector& buffer)const override final
{
typename Vector::const_pointer values_ptr(values);
thrust::gather( m_IndexMap.begin(), m_IndexMap.end(), values_ptr, m_store.data().begin());
m_bijectiveComm.global_scatter_reduce( m_store.data(), thrust::raw_pointer_cast(buffer.data()));
}
virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final
{
typename Vector::pointer values_ptr(values);
if( m_reduction)
{
Vector storet = m_bijectiveComm.global_gather( thrust::raw_pointer_cast(toScatter.data()));
thrust::gather( m_sortMap.begin(), m_sortMap.end(), storet.begin(), m_store.data().begin());
thrust::reduce_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_store.data().begin(), m_keys.data().begin(), values_ptr);
}
else
{
m_bijectiveComm.global_gather( thrust::raw_pointer_cast(toScatter.data()), m_store.data());
thrust::gather( m_sortMap.begin(), m_sortMap.end(), m_store.data().begin(), values_ptr);
}

}
virtual MPI_Comm do_communicator()const override final{return m_bijectiveComm.communicator();}
virtual unsigned do_size() const override final {return m_buffer_size;}
void construct( unsigned local_size, thrust::host_vector<int> localIndexMap, thrust::host_vector<int> pidIndexMap, MPI_Comm comm)
{
this->set_local_size(local_size);
m_bijectiveComm = BijectiveComm<Index, Vector>( pidIndexMap, comm);
m_localIndexMap = localIndexMap, m_pidIndexMap = pidIndexMap;
m_buffer_size = localIndexMap.size();
assert( m_buffer_size == pidIndexMap.size());
Vector m_localIndexMapd = dg::construct<Vector>( localIndexMap);
const typename aCommunicator<Vector>::value_type * v_ptr = thrust::raw_pointer_cast(m_localIndexMapd.data());
Vector gatherMapV = m_bijectiveComm.global_gather( v_ptr); 
m_sortMap = m_sortedIndexMap = m_IndexMap = dg::construct<Index>(gatherMapV);
thrust::sequence( m_sortMap.begin(), m_sortMap.end());
thrust::stable_sort_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_sortMap.begin());
m_store_size = m_IndexMap.size();
m_store.data().resize( m_store_size);
m_keys.data().resize( m_store_size);
Vector temp( m_store_size);
auto new_end = thrust::reduce_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_store.data().begin(), m_keys.data().begin(), temp.begin());
if( new_end.second == temp.end())
m_reduction = false;

}
unsigned m_buffer_size, m_store_size;
BijectiveComm<Index, Vector> m_bijectiveComm;
Index m_IndexMap, m_sortMap, m_sortedIndexMap;
Buffer<Index> m_keys;
Buffer<Vector> m_store;
thrust::host_vector<int> m_localIndexMap, m_pidIndexMap;
bool m_reduction = true;
};


template< class Index, class Vector>
struct GeneralComm : public aCommunicator<Vector>
{
GeneralComm() = default;

GeneralComm( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm) {
construct(local_size, localIndexMap, pidIndexMap, comm);
}


template<class ConversionPolicy>
GeneralComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
{
thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
bool success = true;
for(unsigned i=0; i<local.size(); i++)
if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;
assert( success);
construct(p.local_size(), local, pids, p.communicator());
}

template<class OtherIndex, class OtherVector>
GeneralComm( const GeneralComm<OtherIndex, OtherVector>& src){
if( src.buffer_size() > 0)
construct( src.local_size(), src.getLocalIndexMap(), src.getPidIndexMap(), src.communicator());
}

const thrust::host_vector<int>& getLocalIndexMap() const {return m_surjectiveComm.getLocalIndexMap();}
const thrust::host_vector<int>& getPidIndexMap() const {return m_surjectiveComm.getPidIndexMap();}
virtual GeneralComm* clone() const override final {return new GeneralComm(*this);}
private:
virtual bool do_isCommunicating() const override final{
return m_surjectiveComm.isCommunicating();
}
virtual Vector do_make_buffer() const override final{
Vector tmp(do_size());
return tmp;
}
virtual MPI_Comm do_communicator()const override final{return m_surjectiveComm.communicator();}
virtual void do_global_gather( const get_value_type<Vector>* values, Vector& sink)const override final {
m_surjectiveComm.global_gather( values, sink);
}
virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final {
m_surjectiveComm.global_scatter_reduce( toScatter, thrust::raw_pointer_cast(m_store.data().data()));
typename Vector::pointer values_ptr(values);
dg::blas1::detail::doSubroutine_dispatch(
get_execution_policy<Vector>(),
this->local_size(),
dg::equals(),
0,
values
);
thrust::scatter( m_store.data().begin(), m_store.data().end(), m_scatterMap.begin(), values_ptr);
}

virtual unsigned do_size() const override final{return m_surjectiveComm.buffer_size();}
void construct( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm)
{
this->set_local_size( local_size);
m_surjectiveComm = SurjectiveComm<Index,Vector>(local_size, localIndexMap, pidIndexMap, comm);

const Index& m_sortedIndexMap = m_surjectiveComm.getSortedIndexMap();
thrust::host_vector<int> gatherMap = dg::construct<thrust::host_vector<int>>( m_sortedIndexMap);
thrust::host_vector<int> one( gatherMap.size(), 1), keys(one), number(one);
typedef thrust::host_vector<int>::iterator iterator;
thrust::pair< iterator, iterator> new_end =
thrust::reduce_by_key( gatherMap.begin(), gatherMap.end(), 
one.begin(), keys.begin(), number.begin() );
unsigned distance = thrust::distance( keys.begin(), new_end.first);
m_store.data().resize( distance);
m_scatterMap.resize(distance);
thrust::copy( keys.begin(), keys.begin() + distance, m_scatterMap.begin());
}
SurjectiveComm<Index, Vector> m_surjectiveComm;
Buffer<Vector> m_store;
Index m_scatterMap;
};

}
