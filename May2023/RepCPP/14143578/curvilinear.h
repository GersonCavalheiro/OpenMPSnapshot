#pragma once

#include "dg/algorithm.h"
#include "generator.h"

namespace dg
{
namespace geo
{







struct Topology1d
{
unsigned n;
unsigned N;
bc b = dg::PER;
};


template<class real_type>
struct RealCurvilinearProductGrid3d;
namespace detail
{
template<class host_vector>
dg::SparseTensor<host_vector> square( const dg::SparseTensor<host_vector >& jac, const host_vector& R, bool orthogonal)
{
std::vector<host_vector> values( 5, R);
{
dg::blas1::scal( values[0], 0); 
dg::blas1::pointwiseDot( 1., jac.value(0,0), jac.value(0,0), 1., jac.value(0,1), jac.value(0,1), 0., values[1]); 
dg::blas1::pointwiseDot( 1., jac.value(1,0), jac.value(1,0), 1., jac.value(1,1), jac.value(1,1), 0., values[2]); 
dg::blas1::pointwiseDot( values[3], values[3], values[3]);
dg::blas1::pointwiseDivide( 1., values[3], values[3]); 

dg::blas1::pointwiseDot( 1., jac.value(0,0), jac.value(1,0), 1., jac.value(0,1), jac.value(1,1), 0., values[4]); 
}
SparseTensor<host_vector> metric(values[0]); 
metric.values().pop_back(); 
metric.idx(0,0) = 1; metric.values().push_back( values[1]);
metric.idx(1,1) = 2; metric.values().push_back( values[2]);
metric.idx(2,2) = 3; metric.values().push_back( values[3]);
if( !orthogonal)
{
metric.idx(1,0) = metric.idx(0,1) = 4;
metric.values().push_back( values[4]);
}
return metric;
}
}



template<class real_type>
struct RealCurvilinearGrid2d : public dg::aRealGeometry2d<real_type>
{
RealCurvilinearGrid2d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
RealCurvilinearGrid2d( generator, {n,Nx,bcx}, {n,Ny,bcy}){}
RealCurvilinearGrid2d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty) :
dg::aRealGeometry2d<real_type>( {0, generator.width(), tx.n, tx.N, tx.b}, {0., generator.height(), ty.n, ty.N, ty.b}), m_handle(generator)
{
construct( tx.n, tx.N, ty.n, ty.N);
}


explicit RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g);

const aRealGenerator2d<real_type>& generator() const{return *m_handle;}
virtual RealCurvilinearGrid2d* clone()const override final{return new RealCurvilinearGrid2d(*this);}
private:
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final
{
dg::aRealTopology2d<real_type>::do_set( nx, Nx, ny,Ny);
construct( nx, Nx, ny, Ny);
}
void construct( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny);
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
return m_jac;
}
virtual SparseTensor<thrust::host_vector<real_type>> do_compute_metric( ) const override final{
return m_metric;
}
virtual std::vector<thrust::host_vector<real_type>> do_compute_map()const override final{return m_map;}
dg::SparseTensor<thrust::host_vector<real_type>> m_jac, m_metric;
std::vector<thrust::host_vector<real_type>> m_map;
dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
};



template<class real_type>
struct RealCurvilinearProductGrid3d : public dg::aRealProductGeometry3d<real_type>
{
using perpendicular_grid = RealCurvilinearGrid2d<real_type>;

RealCurvilinearProductGrid3d( const aRealGenerator2d<real_type>& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
RealCurvilinearProductGrid3d( generator, {n,Nx,bcx}, {n,Ny,bcy}, {0., 2.*M_PI, 1,Nz,bcz}){}

RealCurvilinearProductGrid3d( const aRealGenerator2d<real_type>& generator, Topology1d tx, Topology1d ty, RealGrid1d<real_type> gz):
dg::aRealProductGeometry3d<real_type>( {0, generator.width(), tx.n, tx.N, tx.b}, {0., generator.height(), ty.n, ty.N, ty.b},gz), m_handle(generator)
{
m_map.resize(3);
constructPerp( this->nx(), this->Nx(), this->ny(), this->Ny());
constructParallel(this->nz(), this->Nz());
}


const aRealGenerator2d<real_type> & generator() const{return *m_handle;}
virtual RealCurvilinearProductGrid3d* clone()const override final{return new RealCurvilinearProductGrid3d(*this);}
private:
virtual RealCurvilinearGrid2d<real_type>* do_perp_grid() const override final;
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny,unsigned nz,unsigned Nz) override final{
dg::aRealTopology3d<real_type>::do_set( nx, Nx, ny, Ny, nz, Nz);
if( !( nx == this->nx() && Nx == this->Nx() && ny == this->ny() && Ny == this->Ny() ) )
constructPerp( nx, Nx, ny,Ny);
constructParallel(nz,Nz);
}
void constructParallel(unsigned nz,unsigned Nz)
{
m_map[2]=dg::evaluate(dg::cooZ3d, *this);
unsigned size = this->size();
unsigned size2d = this->nx()*this->ny()*this->Nx()*this->Ny();
for( unsigned r=0; r<6;r++)
m_jac.values()[r].resize(size);
m_map[0].resize(size);
m_map[1].resize(size);
for( unsigned k=1; k<nz*Nz; k++)
for( unsigned i=0; i<size2d; i++)
{
for(unsigned r=0; r<6; r++)
m_jac.values()[r][k*size2d+i] = m_jac.values()[r][(k-1)*size2d+i];
m_map[0][k*size2d+i] = m_map[0][(k-1)*size2d+i];
m_map[1][k*size2d+i] = m_map[1][(k-1)*size2d+i];
}
}
void constructPerp( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)
{
dg::Grid1d gX1d( this->x0(), this->x1(), nx, Nx);
dg::Grid1d gY1d( this->y0(), this->y1(), ny, Ny);
thrust::host_vector<real_type> x_vec = dg::evaluate( dg::cooX1d, gX1d);
thrust::host_vector<real_type> y_vec = dg::evaluate( dg::cooX1d, gY1d);
m_jac = SparseTensor< thrust::host_vector<real_type>>( x_vec);
m_jac.values().resize( 6);
m_handle->generate( x_vec, y_vec, m_map[0], m_map[1], m_jac.values()[2], m_jac.values()[3], m_jac.values()[4], m_jac.values()[5]);
m_jac.idx(0,0) = 2, m_jac.idx(0,1) = 3, m_jac.idx(1,0)=4, m_jac.idx(1,1) = 5;
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian( ) const override final{
return m_jac;
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric( ) const override final
{
return detail::square( m_jac, m_map[0], m_handle->isOrthogonal());
}
virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{return m_map;}
std::vector<thrust::host_vector<real_type> > m_map;
SparseTensor<thrust::host_vector<real_type> > m_jac;
dg::ClonePtr<aRealGenerator2d<real_type>> m_handle;
};

using CurvilinearGrid2d         = dg::geo::RealCurvilinearGrid2d<double>;
using CurvilinearProductGrid3d  = dg::geo::RealCurvilinearProductGrid3d<double>;
#ifndef MPI_VERSION
namespace x{
using CurvilinearGrid2d         = CurvilinearGrid2d        ;
using CurvilinearProductGrid3d  = CurvilinearProductGrid3d ;
}
#endif

template<class real_type>
RealCurvilinearGrid2d<real_type>::RealCurvilinearGrid2d( RealCurvilinearProductGrid3d<real_type> g):
dg::aRealGeometry2d<real_type>( g.gx(), g.gy() ), m_handle(g.generator())
{
g.set( this->nx(), this->Nx(), this->ny(), this->Ny(), 1, 1); 
m_map=g.map();
m_jac=g.jacobian();
m_metric=g.metric();
dg::blas1::copy( 1., m_metric.values()[3]);
m_map.pop_back();
}
template<class real_type>
void RealCurvilinearGrid2d<real_type>::construct( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny)
{
RealCurvilinearProductGrid3d<real_type> g( *m_handle, {nx,Nx,this->bcx()}, {ny,Ny,this->bcy()}, {0., 2.*M_PI, 1,1});
*this = RealCurvilinearGrid2d<real_type>(g);
}
template<class real_type>
typename RealCurvilinearProductGrid3d<real_type>::perpendicular_grid* RealCurvilinearProductGrid3d<real_type>::do_perp_grid() const { return new typename RealCurvilinearProductGrid3d<real_type>::perpendicular_grid(*this);}

}
}
