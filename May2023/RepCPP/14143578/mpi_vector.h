#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exceptions.h"
#include "exblas/mpi_accumulate.h"
#include "tensor_traits.h"
#include "blas1_dispatch_shared.h"
#include "mpi_communicator.h"
#include "memory.h"
#include "config.h"

namespace dg
{


template<class container>
struct MPI_Vector
{
typedef container container_type;
MPI_Vector(){
m_comm = m_comm128 = m_comm128Reduce = MPI_COMM_NULL;
}

MPI_Vector( const container& data, MPI_Comm comm): m_data( data), m_comm(comm) {
exblas::mpi_reduce_communicator( comm, &m_comm128, &m_comm128Reduce);
}


template<class OtherContainer>
MPI_Vector( const MPI_Vector<OtherContainer>& src){
m_data = src.data();
m_comm = src.communicator();
m_comm128 = src.communicator_mod();
m_comm128Reduce = src.communicator_mod_reduce();
}

const container& data() const {return m_data;}
container& data() {return m_data;}

MPI_Comm communicator() const{return m_comm;}
MPI_Comm communicator_mod() const{return m_comm128;}


MPI_Comm communicator_mod_reduce() const{return m_comm128Reduce;}


void set_communicator(MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce){
m_comm = comm;
m_comm128 = comm_mod;
m_comm128Reduce = comm_mod_reduce;
}

unsigned size() const{return m_data.size();}

void swap( MPI_Vector& src){
m_data.swap(src.m_data);
std::swap( m_comm , src.m_comm);
std::swap( m_comm128 , src.m_comm128);
std::swap( m_comm128Reduce , src.m_comm128Reduce);
}
private:
container m_data;
MPI_Comm m_comm, m_comm128, m_comm128Reduce;

};
template<class container>
void swap( MPI_Vector<container>& a, MPI_Vector<container>& b){
a.swap(b);
}


template<class container>
struct TensorTraits<MPI_Vector<container> > {
using value_type = get_value_type<container>;
using tensor_category = MPIVectorTag;
using execution_policy = get_execution_policy<container>;
};


template<class Index, class Buffer, class Vector>
struct NearestNeighborComm
{
using container_type = Vector;
using buffer_type = Buffer;
using pointer_type = get_value_type<Vector>*;
using const_pointer_type = get_value_type<Vector> const *;
NearestNeighborComm( MPI_Comm comm = MPI_COMM_NULL){
m_comm = comm;
m_silent = true;
}

NearestNeighborComm( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction)
{
static_assert( std::is_same<const_pointer_type, get_value_type<Buffer>>::value, "Must be same pointer types");
construct( n, vector_dimensions, comm, direction);
}


template< class OtherIndex, class OtherBuffer, class OtherVector>
NearestNeighborComm( const NearestNeighborComm<OtherIndex, OtherBuffer, OtherVector>& src){
if( src.buffer_size() == 0)  m_silent=true;
else
construct( src.n(), src.dims(), src.communicator(), src.direction());
}


unsigned n() const{return m_n;}

const unsigned* dims() const{return m_dim;}

unsigned direction() const {return m_direction;}
MPI_Comm communicator() const{return m_comm;}


Buffer allocate_buffer( )const{
if( buffer_size() == 0 ) return Buffer();
return Buffer(6);
}

unsigned buffer_size() const;
bool isCommunicating() const{
if( buffer_size() == 0) return false;
return true;
}

int map_index(int i) const{
if( i==-1) return 0;
if( i== 0) return 1;
if( i==+1) return 2;
if( i==(int)m_outer_size-0) return 5;
if( i==(int)m_outer_size-1) return 4;
if( i==(int)m_outer_size-2) return 3;
throw Error( Message(_ping_)<<"Index not mappable!");
return -1;
}


void global_gather_init( const_pointer_type input, buffer_type& buffer, MPI_Request rqst[4])const
{
unsigned size = buffer_size();
const_pointer_type host_ptr[6];
if(m_trivial)
{
host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]);
host_ptr[1] = input;
host_ptr[2] = input+size;
host_ptr[3] = input+(m_outer_size-2)*size;
host_ptr[4] = input+(m_outer_size-1)*size;
host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]);
}
else
{
host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]);
host_ptr[1] = thrust::raw_pointer_cast(&m_internal_buffer.data()[1*size]);
host_ptr[2] = thrust::raw_pointer_cast(&m_internal_buffer.data()[2*size]);
host_ptr[3] = thrust::raw_pointer_cast(&m_internal_buffer.data()[3*size]);
host_ptr[4] = thrust::raw_pointer_cast(&m_internal_buffer.data()[4*size]);
host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]);
}
thrust::copy( host_ptr, host_ptr+6, buffer.begin());
do_global_gather_init( get_execution_policy<Vector>(), input, rqst);
sendrecv( host_ptr[1], host_ptr[4],
thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]), 
thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]), 
rqst);
}

void global_gather_wait(const_pointer_type input, const buffer_type& buffer, MPI_Request rqst[4])const
{
MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
#ifdef _DG_CUDA_UNAWARE_MPI
if( std::is_same< get_execution_policy<Vector>, CudaTag>::value ) 
{
unsigned size = buffer_size();
cudaError_t code = cudaGetLastError( );
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]), 
thrust::raw_pointer_cast(&m_internal_host_buffer.data()[0*size]), 
size*sizeof(get_value_type<Vector>), cudaMemcpyHostToDevice);
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));

code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]), 
thrust::raw_pointer_cast(&m_internal_host_buffer.data()[5*size]), 
size*sizeof(get_value_type<Vector>), cudaMemcpyHostToDevice);
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif
}
private:
void do_global_gather_init( OmpTag, const_pointer_type, MPI_Request rqst[4])const;
void do_global_gather_init( SerialTag, const_pointer_type, MPI_Request rqst[4])const;
void do_global_gather_init( CudaTag, const_pointer_type, MPI_Request rqst[4])const;
void construct( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction);

unsigned m_n, m_dim[3]; 
MPI_Comm m_comm;
unsigned m_direction;
bool m_silent, m_trivial=false; 
unsigned m_outer_size = 1; 
Index m_gather_map_middle;
dg::Buffer<Vector> m_internal_buffer;
#ifdef _DG_CUDA_UNAWARE_MPI
dg::Buffer<thrust::host_vector<get_value_type<Vector>>> m_internal_host_buffer;
#endif

void sendrecv(const_pointer_type, const_pointer_type, pointer_type, pointer_type, MPI_Request rqst[4])const;
int m_source[2], m_dest[2];
};


template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::construct( unsigned n, const unsigned dimensions[3], MPI_Comm comm, unsigned direction)
{
static_assert( std::is_base_of<SharedVectorTag, get_tensor_category<V>>::value,
"Only Shared vectors allowed");
m_silent=false;
m_n=n;
m_dim[0] = dimensions[0], m_dim[1] = dimensions[1], m_dim[2] = dimensions[2];
m_direction = direction;
if( dimensions[2] == 1 && direction == 1) m_trivial = true;
else if( direction == 2) m_trivial = true;
else m_trivial = false;
assert( direction <3);
m_comm = comm;
MPI_Cart_shift( m_comm, m_direction, -1, &m_source[0], &m_dest[0]);
MPI_Cart_shift( m_comm, m_direction, +1, &m_source[1], &m_dest[1]);
{
int ndims;
MPI_Cartdim_get( comm, &ndims);
int dims[ndims], periods[ndims], coords[ndims];
MPI_Cart_get( comm, ndims, dims, periods, coords);
if( dims[direction] == 1) m_silent = true;
}
if( !m_silent)
{
m_outer_size = dimensions[0]*dimensions[1]*dimensions[2]/buffer_size();
assert( m_outer_size > 1 && "Parallelization too fine grained!"); 
thrust::host_vector<int> mid_gather( 4*buffer_size());
switch( direction)
{
case( 0):
for( unsigned i=0; i<m_dim[2]*m_dim[1]; i++)
for( unsigned j=0; j<n; j++)
{
mid_gather[(0*n+j)*m_dim[2]*m_dim[1]+i] = i*m_dim[0]                + j;
mid_gather[(1*n+j)*m_dim[2]*m_dim[1]+i] = i*m_dim[0] + n            + j;
mid_gather[(2*n+j)*m_dim[2]*m_dim[1]+i] = i*m_dim[0] + m_dim[0]-2*n + j;
mid_gather[(3*n+j)*m_dim[2]*m_dim[1]+i] = i*m_dim[0] + m_dim[0]-  n + j;
}
break;
case( 1):
for( unsigned i=0; i<m_dim[2]; i++)
for( unsigned j=0; j<n; j++)
for( unsigned k=0; k<m_dim[0]; k++)
{
mid_gather[((0*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1]                + j)*m_dim[0] + k;
mid_gather[((1*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + n            + j)*m_dim[0] + k;
mid_gather[((2*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + m_dim[1]-2*n + j)*m_dim[0] + k;
mid_gather[((3*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + m_dim[1]-  n + j)*m_dim[0] + k;
}
break;
case( 2):
for( unsigned i=0; i<n; i++)
for( unsigned j=0; j<m_dim[0]*m_dim[1]; j++)
{
mid_gather[(0*n+i)*m_dim[0]*m_dim[1]+j] = (i                )*m_dim[0]*m_dim[1] + j;
mid_gather[(1*n+i)*m_dim[0]*m_dim[1]+j] = (i + n            )*m_dim[0]*m_dim[1] + j;
mid_gather[(2*n+i)*m_dim[0]*m_dim[1]+j] = (i + m_dim[2]-2*n )*m_dim[0]*m_dim[1] + j;
mid_gather[(3*n+i)*m_dim[0]*m_dim[1]+j] = (i + m_dim[2]-  n )*m_dim[0]*m_dim[1] + j;
}
break;
}
m_gather_map_middle = mid_gather; 
m_internal_buffer.data().resize( 6*buffer_size() );
#ifdef _DG_CUDA_UNAWARE_MPI
m_internal_host_buffer.data().resize( 6*buffer_size() );
#endif
}
}

template<class I, class B, class V>
unsigned NearestNeighborComm<I,B,V>::buffer_size() const
{
if( m_silent) return 0;
switch( m_direction)
{
case( 0): 
return m_n*m_dim[1]*m_dim[2];
case( 1): 
return m_n*m_dim[0]*m_dim[2];
case( 2): 
return m_n*m_dim[0]*m_dim[1]; 
default:
return 0;
}
}

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( SerialTag, const_pointer_type input, MPI_Request rqst[4]) const
{
if( !m_trivial)
{
unsigned size = buffer_size();
for( unsigned i=0; i<4*size; i++)
m_internal_buffer.data()[i+size] = input[m_gather_map_middle[i]];
}
}
#ifdef _OPENMP
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( OmpTag, const_pointer_type input, MPI_Request rqst[4]) const
{
if(!m_trivial)
{
unsigned size = buffer_size();
#pragma omp parallel for
for( unsigned i=0; i<4*size; i++)
m_internal_buffer.data()[size+i] = input[m_gather_map_middle[i]];
}
}
#endif
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( CudaTag, const_pointer_type input, MPI_Request rqst[4]) const
{
if(!m_trivial)
{
unsigned size = buffer_size();
thrust::gather( thrust::cuda::tag(), m_gather_map_middle.begin(), m_gather_map_middle.end(), input, m_internal_buffer.data().begin()+size);
}
cudaError_t code = cudaGetLastError( );
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaDeviceSynchronize(); 
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::sendrecv( const_pointer_type sb1_ptr, const_pointer_type sb2_ptr, pointer_type rb1_ptr, pointer_type rb2_ptr, MPI_Request rqst[4]) const
{
unsigned size = buffer_size();
#ifdef _DG_CUDA_UNAWARE_MPI
if( std::is_same< get_execution_policy<V>, CudaTag>::value ) 
{
cudaError_t code = cudaGetLastError( );
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]),
sb1_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); 
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]),  
sb2_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); 
if( code != cudaSuccess)
throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
sb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]);
sb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]);
rb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[0*size]);
rb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[5*size]);
}
#endif
MPI_Isend( sb1_ptr, size,
getMPIDataType<get_value_type<V>>(),  
m_dest[0], 3, m_comm, &rqst[0]); 
MPI_Irecv( rb2_ptr, size,
getMPIDataType<get_value_type<V>>(), 
m_source[0], 3, m_comm, &rqst[1]); 

MPI_Isend( sb2_ptr, size,
getMPIDataType<get_value_type<V>>(),  
m_dest[1], 9, m_comm, &rqst[2]);  
MPI_Irecv( rb1_ptr, size,
getMPIDataType<get_value_type<V>>(), 
m_source[1], 9, m_comm, &rqst[3]); 
}


}
