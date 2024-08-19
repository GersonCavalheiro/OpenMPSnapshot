#pragma once

#include <mpi.h>

#include "dg/algorithm.h"
#include "curvilinear.h"
#include "generator.h"

namespace dg
{
namespace geo
{

template<class real_type>
struct RealCurvilinearProductMPIGrid3d;

template<class real_type>
struct RealCurvilinearMPIGrid2d : public dg::aRealMPIGeometry2d<real_type>
{
RealCurvilinearMPIGrid2d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm comm):
RealCurvilinearMPIGrid2d( generator, {n,Nx,bcx}, {n,Ny,bcy}, comm){}

RealCurvilinearMPIGrid2d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty, MPI_Comm comm):
dg::aRealMPIGeometry2d<real_type>( {0, generator.width(), tx.n, tx.N, tx.b}, {0., generator.height(), ty.n, ty.N, ty.b}, comm), m_handle(generator)
{
RealCurvilinearGrid2d<real_type> g(generator, tx, ty);
divide_and_conquer(g);
}
explicit RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g);

const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
virtual RealCurvilinearMPIGrid2d* clone()const override final{return new RealCurvilinearMPIGrid2d(*this);}
virtual RealCurvilinearGrid2d<real_type>* global_geometry()const override final{
return new RealCurvilinearGrid2d<real_type>(
*m_handle,
{global().nx(), global().Nx(), global().bcx()},
{global().ny(), global().Ny(), global().bcy()});
}
using dg::aRealMPIGeometry2d<real_type>::global;
private:
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final
{
dg::aRealMPITopology2d<real_type>::do_set(nx, Nx, ny, Ny);
RealCurvilinearGrid2d<real_type> g( *m_handle, {nx, Nx}, {ny, Ny});
divide_and_conquer(g);
}
void divide_and_conquer(const RealCurvilinearGrid2d<real_type>& g_)
{
dg::SparseTensor<thrust::host_vector<real_type> > jacobian=g_.jacobian();
dg::SparseTensor<thrust::host_vector<real_type> > metric=g_.metric();
std::vector<thrust::host_vector<real_type> > map = g_.map();
for( unsigned i=0; i<3; i++)
for( unsigned j=0; j<3; j++)
{
m_metric.idx(i,j) = metric.idx(i,j);
m_jac.idx(i,j) = jacobian.idx(i,j);
}
m_jac.values().resize( jacobian.values().size());
for( unsigned i=0; i<jacobian.values().size(); i++)
m_jac.values()[i] = global2local( jacobian.values()[i], *this);
m_metric.values().resize( metric.values().size());
for( unsigned i=0; i<metric.values().size(); i++)
m_metric.values()[i] = global2local( metric.values()[i], *this);
m_map.resize(map.size());
for( unsigned i=0; i<map.size(); i++)
m_map[i] = global2local( map[i], *this);
}

virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_jacobian( ) const override final{
return m_jac;
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_metric( ) const override final{
return m_metric;
}
virtual std::vector<MPI_Vector<thrust::host_vector<real_type>>> do_compute_map()const override final{return m_map;}
dg::SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> m_jac, m_metric;
std::vector<MPI_Vector<thrust::host_vector<real_type>>> m_map;
dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
};


template<class real_type>
struct RealCurvilinearProductMPIGrid3d : public dg::aRealProductMPIGeometry3d<real_type>
{
typedef dg::geo::RealCurvilinearMPIGrid2d<real_type> perpendicular_grid; 
RealCurvilinearProductMPIGrid3d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
RealCurvilinearProductMPIGrid3d( generator, {n,Nx,bcx}, {n,Ny,bcy}, {0.,2.*M_PI,1,Nz,bcz}, comm){}


RealCurvilinearProductMPIGrid3d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty, RealGrid1d<real_type> gz, MPI_Comm comm):
dg::aRealProductMPIGeometry3d<real_type>( {0, generator.width(), tx.n, tx.N, tx.b}, {0., generator.height(), ty.n, ty.N, ty.b}, gz, comm),
m_handle( generator)
{
m_map.resize(3);
RealCurvilinearMPIGrid2d<real_type> g(generator,tx,ty,this->get_perp_comm());
constructPerp( g);
constructParallel(this->nz(), this->local().Nz());
}


const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
virtual RealCurvilinearProductMPIGrid3d* clone()const override final{return new RealCurvilinearProductMPIGrid3d(*this);}
virtual RealCurvilinearProductGrid3d<real_type>* global_geometry()const{
return new RealCurvilinearProductGrid3d<real_type>(
*m_handle,
{global().nx(), global().Nx(), global().bcx()},
{global().ny(), global().Ny(), global().bcy()},
{global().x0(), global().x1(), global().nz(), global().Nz(), global().bcz()});
}
using dg::aRealMPIGeometry3d<real_type>::global;
private:
virtual perpendicular_grid* do_perp_grid() const override final{ return new perpendicular_grid(*this);}
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final
{
dg::aRealMPITopology3d<real_type>::do_set(nx, Nx, ny, Ny, nz, Nz);
if( !( nx == this->nx() && Nx == global().Nx() && ny == this->ny() && Ny == global().Ny() ) )
{
RealCurvilinearMPIGrid2d<real_type> g( *m_handle,{nx,Nx,this->bcx()},{ny,Ny, this->bcy()}, this->get_perp_comm());
constructPerp( g);
}
constructParallel(this->nz(), this->local().Nz());
}
void constructPerp( RealCurvilinearMPIGrid2d<real_type>& g2d)
{
m_jac=g2d.jacobian();
m_map=g2d.map();
}
void constructParallel( unsigned nz, unsigned localNz )
{
m_map.resize(3);
m_map[2]=dg::evaluate(dg::cooZ3d, *this);
unsigned size = this->local().size();
unsigned size2d = this->nx()*this->ny()*this->local().Nx()*this->local().Ny();
MPI_Comm comm = this->communicator(), comm_mod, comm_mod_reduce;
exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
for( unsigned r=0; r<6;r++)
{
m_jac.values()[r].data().resize(size);
m_jac.values()[r].set_communicator( comm, comm_mod, comm_mod_reduce);
}
m_map[0].data().resize(size);
m_map[0].set_communicator( comm, comm_mod, comm_mod_reduce);
m_map[1].data().resize(size);
m_map[1].set_communicator( comm, comm_mod, comm_mod_reduce);
for( unsigned k=1; k<nz*localNz; k++)
for( unsigned i=0; i<size2d; i++)
{
for(unsigned r=0; r<6; r++)
m_jac.values()[r].data()[k*size2d+i] = m_jac.values()[r].data()[(k-1)*size2d+i];
m_map[0].data()[k*size2d+i] = m_map[0].data()[(k-1)*size2d+i];
m_map[1].data()[k*size2d+i] = m_map[1].data()[(k-1)*size2d+i];
}
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_jacobian( ) const override final{
return m_jac;
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> do_compute_metric( ) const override final{
return detail::square( m_jac, m_map[0], m_handle->isOrthogonal());
}
virtual std::vector<MPI_Vector<thrust::host_vector<real_type>>> do_compute_map()const override final{return m_map;}
dg::SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> m_jac;
std::vector<MPI_Vector<thrust::host_vector<real_type>>> m_map;
ClonePtr<dg::geo::aRealGenerator2d<real_type>> m_handle;
};
template<class real_type>
RealCurvilinearMPIGrid2d<real_type>::RealCurvilinearMPIGrid2d( const RealCurvilinearProductMPIGrid3d<real_type>& g):
dg::aRealMPIGeometry2d<real_type>( g.global().gx(), g.global().gy(), g.get_perp_comm() ),
m_handle(g.generator())
{
m_map=g.map();
m_jac=g.jacobian();
m_metric=g.metric();
m_map.pop_back();
unsigned s = this->local().size();
MPI_Comm comm = g.get_perp_comm(), comm_mod, comm_mod_reduce;
exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
for( unsigned i=0; i<m_jac.values().size(); i++)
{
m_jac.values()[i].data().resize(s);
m_jac.values()[i].set_communicator( comm, comm_mod, comm_mod_reduce);
}
for( unsigned i=0; i<m_metric.values().size(); i++)
{
m_metric.values()[i].data().resize(s);
m_metric.values()[i].set_communicator( comm, comm_mod, comm_mod_reduce);
}
dg::blas1::copy( 1., m_metric.values()[3]);
for( unsigned i=0; i<m_map.size(); i++)
{
m_map[i].data().resize(s);
m_map[i].set_communicator( comm, comm_mod, comm_mod_reduce);
}
}
using CurvilinearMPIGrid2d         = dg::geo::RealCurvilinearMPIGrid2d<double>;
using CurvilinearProductMPIGrid3d  = dg::geo::RealCurvilinearProductMPIGrid3d<double>;
namespace x{
using CurvilinearGrid2d         = CurvilinearMPIGrid2d        ;
using CurvilinearProductGrid3d  = CurvilinearProductMPIGrid3d ;
}

}
}

