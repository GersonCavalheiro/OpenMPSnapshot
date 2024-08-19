#pragma once

#include "../blas2.h"
#include "average_cpu.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "average_gpu.cuh"
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include "average_omp.h"
#endif

namespace dg{


template<class ContainerType>
void transpose( unsigned nx, unsigned ny, const ContainerType& in, ContainerType& out)
{
assert(&in != &out);
using value_type = get_value_type<ContainerType>;
dg::blas2::parallel_for( [nx,ny]DG_DEVICE( unsigned k, const value_type* ii, value_type* oo)
{
unsigned i = k/nx, j =  k%nx;
oo[j*ny+i] = ii[i*nx+j];
}, nx*ny, in, out);
}


template<class ContainerType>
void extend_line( unsigned nx, unsigned ny, const ContainerType& in, ContainerType& out)
{
assert(&in != &out);
using value_type = get_value_type<ContainerType>;
dg::blas2::parallel_for( [nx]DG_DEVICE( unsigned k, const value_type* ii, value_type* oo)
{
unsigned i = k/nx, j =  k%nx;
oo[i*nx+j] = ii[j];
}, nx*ny, in, out);
}

template<class ContainerType>
void extend_column( unsigned nx, unsigned ny, const ContainerType& in, ContainerType& out)
{
assert(&in != &out);
using value_type = get_value_type<ContainerType>;
dg::blas2::parallel_for( [nx]DG_DEVICE( unsigned k, const value_type* ii, value_type* oo)
{
unsigned i = k/nx, j =  k%nx;
oo[i*nx+j] = ii[i];
}, nx*ny, in, out);
}

template<class ContainerType>
void average( unsigned nx, unsigned ny, const ContainerType& in0, const ContainerType& in1, ContainerType& out)
{
static_assert( std::is_same<get_value_type<ContainerType>, double>::value, "We only support double precision dot products at the moment!");
const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
double* out_ptr = thrust::raw_pointer_cast( out.data());
average( get_execution_policy<ContainerType>(), nx, ny, in0_ptr, in1_ptr, out_ptr);
}

#ifdef MPI_VERSION
template<class ContainerType>
void mpi_average( unsigned nx, unsigned ny, const ContainerType& in0, const ContainerType& in1, ContainerType& out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce)
{
static_assert( std::is_same<get_value_type<ContainerType>, double>::value, "We only support double precision dot products at the moment!");
const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
double* out_ptr = thrust::raw_pointer_cast( out.data());
average_mpi( get_execution_policy<ContainerType>(), nx, ny, in0_ptr, in1_ptr, out_ptr, comm, comm_mod, comm_mod_reduce);
}
#endif 

}
