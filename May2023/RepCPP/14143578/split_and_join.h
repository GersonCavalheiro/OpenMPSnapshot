#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "dg/backend/blas1_dispatch_shared.h"
#include "dg/backend/view.h"
#include "dg/blas1.h"
#include "grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "mpi_evaluation.h"
#endif 

namespace dg
{



template<class SharedContainer, class real_type>
void split( SharedContainer& in, std::vector<View<SharedContainer>>& out, const aRealTopology3d<real_type>& grid)
{
assert( out.size() == grid.nz()*grid.Nz());
unsigned size2d=grid.nx()*grid.ny()*grid.Nx()*grid.Ny();
for(unsigned i=0; i<grid.nz()*grid.Nz(); i++)
out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
}


template<class SharedContainer, class real_type>
std::vector<View<SharedContainer>> split( SharedContainer& in, const aRealTopology3d<real_type>& grid)
{
std::vector<View<SharedContainer>> out;
RealGrid3d<real_type> l( grid);
unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
out.resize( l.nz()*l.Nz());
for(unsigned i=0; i<l.nz()*l.Nz(); i++)
out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
return out;
}


template<class Container, class real_type>
void assign3dfrom2d( const thrust::host_vector<real_type>& in2d, Container&
out, const aRealTopology3d<real_type>& grid)
{
thrust::host_vector<real_type> vector( grid.size());
std::vector<dg::View< thrust::host_vector<real_type>>> view =
dg::split( vector, grid); 
for( unsigned i=0; i<grid.nz()*grid.Nz(); i++)
dg::blas1::copy( in2d, view[i]);
dg::assign( vector, out);
}


#ifdef MPI_VERSION

template<class MPIContainer>
using get_mpi_view_type =
std::conditional_t< std::is_const<MPIContainer>::value,
MPI_Vector<View<const typename MPIContainer::container_type>>,
MPI_Vector<View<typename MPIContainer::container_type>> >;


template<class MPIContainer, class real_type>
void split( MPIContainer& in, std::vector<get_mpi_view_type<MPIContainer> >&
out, const aRealMPITopology3d<real_type>& grid)
{
RealGrid3d<real_type> l = grid.local();
unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
for(unsigned i=0; i<l.nz()*l.Nz(); i++)
{
out[i].data().construct( thrust::raw_pointer_cast(in.data().data()) +
i*size2d, size2d);
}
}

template< class MPIContainer, class real_type>
std::vector<get_mpi_view_type<MPIContainer> > split(
MPIContainer& in, const aRealMPITopology3d<real_type>& grid)
{
std::vector<get_mpi_view_type<MPIContainer>> out;
int result;
MPI_Comm_compare( in.communicator(), grid.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm planeComm = grid.get_perp_comm(), comm_mod, comm_mod_reduce;
exblas::mpi_reduce_communicator( planeComm, &comm_mod, &comm_mod_reduce);
RealGrid3d<real_type> l = grid.local();
unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
out.resize( l.nz()*l.Nz());
for(unsigned i=0; i<l.nz()*l.Nz(); i++)
{
out[i].data().construct( thrust::raw_pointer_cast(in.data().data())
+ i*size2d, size2d);
out[i].set_communicator( planeComm, comm_mod, comm_mod_reduce);
}
return out;
}


template<class LocalContainer, class real_type>
void assign3dfrom2d( const MPI_Vector<thrust::host_vector<real_type>>& in2d,
MPI_Vector<LocalContainer>& out,
const aRealMPITopology3d<real_type>& grid)
{
MPI_Vector<thrust::host_vector<real_type>> vector = dg::evaluate( dg::zero, grid);
std::vector<MPI_Vector<dg::View<thrust::host_vector<real_type>>> > view =
dg::split( vector, grid); 
for( unsigned i=0; i<grid.nz()*grid.local().Nz(); i++)
dg::blas1::copy( in2d, view[i]);
dg::assign( vector, out);
}
#endif 

}
