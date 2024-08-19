#pragma once

#include "gridX.h"
#include "evaluationX.h"
#include "tensor.h"

namespace dg
{



template<class real_type>
struct aRealGeometryX2d : public aRealTopologyX2d<real_type>
{
SparseTensor<thrust::host_vector<real_type> > jacobian()const{
return do_compute_jacobian();
}
SparseTensor<thrust::host_vector<real_type> > metric()const {
return do_compute_metric();
}
std::vector<thrust::host_vector<real_type> > map()const{
return do_compute_map();
}
virtual aRealGeometryX2d* clone()const=0;
virtual ~aRealGeometryX2d() = default;
protected:
using aRealTopologyX2d<real_type>::aRealTopologyX2d;
aRealGeometryX2d( const aRealGeometryX2d& src) = default;
aRealGeometryX2d& operator=( const aRealGeometryX2d& src) = default;
private:
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
return SparseTensor<thrust::host_vector<real_type> >(*this);
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
return SparseTensor<thrust::host_vector<real_type> >(*this);
}
virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
std::vector<thrust::host_vector<real_type> > map(2);
map[0] = dg::evaluate(dg::cooX2d, *this);
map[1] = dg::evaluate(dg::cooY2d, *this);
return map;
}


};


template<class real_type>
struct aRealGeometryX3d : public aRealTopologyX3d<real_type>
{
SparseTensor<thrust::host_vector<real_type> > jacobian()const{
return do_compute_jacobian();
}
SparseTensor<thrust::host_vector<real_type> > metric()const {
return do_compute_metric();
}
std::vector<thrust::host_vector<real_type> > map()const{
return do_compute_map();
}
virtual aRealGeometryX3d* clone()const=0;
virtual ~aRealGeometryX3d() = default;
protected:
using aRealTopologyX3d<real_type>::aRealTopologyX3d;
aRealGeometryX3d( const aRealGeometryX3d& src) = default;
aRealGeometryX3d& operator=( const aRealGeometryX3d& src) = default;
private:
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
return SparseTensor<thrust::host_vector<real_type> >();
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
return SparseTensor<thrust::host_vector<real_type> >();
}
virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
std::vector<thrust::host_vector<real_type> > map(3);
map[0] = dg::evaluate(dg::cooX3d, *this);
map[1] = dg::evaluate(dg::cooY3d, *this);
map[2] = dg::evaluate(dg::cooZ3d, *this);
return map;
}
};




template<class real_type>
struct RealCartesianGridX2d: public dg::aRealGeometryX2d<real_type>
{
RealCartesianGridX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aRealGeometryX2d<real_type>(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}

RealCartesianGridX2d( const dg::GridX2d& g):dg::aRealGeometryX2d<real_type>(g.x0(),g.x1(),g.y0(),g.y1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
virtual RealCartesianGridX2d* clone()const override final{
return new RealCartesianGridX2d(*this);
}
};


template<class real_type>
struct RealCartesianGridX3d: public dg::aRealGeometryX3d<real_type>
{
RealCartesianGridX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealGeometryX3d<real_type>(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}

RealCartesianGridX3d( const dg::GridX3d& g):dg::aRealGeometryX3d<real_type>(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
virtual RealCartesianGridX3d* clone()const override final{
return new RealCartesianGridX3d(*this);
}
};

template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aRealGeometryX2d<real_type>& g)
{
std::vector<thrust::host_vector<real_type> > map = g.map();
thrust::host_vector<real_type> vec( g.size());
for( unsigned i=0; i<g.size(); i++)
vec[i] = f( map[0][i], map[1][i]);
return vec;
}

template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aRealGeometryX3d<real_type>& g)
{
std::vector<thrust::host_vector<real_type> > map = g.map();
thrust::host_vector<real_type> vec( g.size());
for( unsigned i=0; i<g.size(); i++)
vec[i] = f( map[0][i], map[1][i], map[2][i]);
return vec;
}

using CartesianGridX2d  = dg::RealCartesianGridX2d<double>;
using CartesianGridX3d  = dg::RealCartesianGridX3d<double>;
using aGeometryX2d      = dg::aRealGeometryX2d<double>;
using aGeometryX3d      = dg::aRealGeometryX3d<double>;
} 
