#pragma once

#include "grid.h"
#include "tensor.h"

namespace dg
{


template<class real_type>
struct aRealGeometry2d : public aRealTopology2d<real_type>
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
virtual aRealGeometry2d* clone()const=0;
virtual ~aRealGeometry2d() = default;
protected:
using aRealTopology2d<real_type>::aRealTopology2d;
aRealGeometry2d( const aRealGeometry2d& src) = default;
aRealGeometry2d& operator=( const aRealGeometry2d& src) = default;
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
struct aRealGeometry3d : public aRealTopology3d<real_type>
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
virtual aRealGeometry3d* clone()const=0;
virtual ~aRealGeometry3d() = default;
protected:
using aRealTopology3d<real_type>::aRealTopology3d;
aRealGeometry3d( const aRealGeometry3d& src) = default;
aRealGeometry3d& operator=( const aRealGeometry3d& src) = default;
private:
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
return SparseTensor<thrust::host_vector<real_type> >(*this);
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
return SparseTensor<thrust::host_vector<real_type> >(*this);
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
struct aRealProductGeometry3d : public aRealGeometry3d<real_type>
{

aRealGeometry2d<real_type>* perp_grid()const{
return do_perp_grid();
}
virtual ~aRealProductGeometry3d() = default;
virtual aRealProductGeometry3d* clone()const=0;
protected:
using aRealGeometry3d<real_type>::aRealGeometry3d;
aRealProductGeometry3d( const aRealProductGeometry3d& src) = default;
aRealProductGeometry3d& operator=( const aRealProductGeometry3d& src) = default;
private:
virtual aRealGeometry2d<real_type>* do_perp_grid()const=0;
};



template<class real_type>
struct RealCartesianGrid2d: public dg::aRealGeometry2d<real_type>
{
RealCartesianGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aRealGeometry2d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy}){}

RealCartesianGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): dg::aRealGeometry2d<real_type>(gx,gy){}

RealCartesianGrid2d( const dg::RealGrid2d<real_type>& g):dg::aRealGeometry2d<real_type>(g.gx(), g.gy()){}
virtual RealCartesianGrid2d<real_type>* clone()const override final{
return new RealCartesianGrid2d<real_type>(*this);
}
private:
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
aRealTopology2d<real_type>::do_set(nx,Nx,ny,Ny);
}
};


template<class real_type>
struct RealCartesianGrid3d: public dg::aRealProductGeometry3d<real_type>
{
using perpendicular_grid = RealCartesianGrid2d<real_type>;
RealCartesianGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>({x0,x1,n,Nx,bcx}, {y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}){}

RealCartesianGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): dg::aRealProductGeometry3d<real_type>(gx,gy,gz){}

RealCartesianGrid3d( const dg::RealGrid3d<real_type>& g):dg::aRealProductGeometry3d<real_type>(g.gx(), g.gy(), g.gz()){}
virtual RealCartesianGrid3d* clone()const override final{
return new RealCartesianGrid3d(*this);
}
private:
virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
return new RealCartesianGrid2d<real_type>(this->gx(), this->gy());
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final{
aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};


template<class real_type>
struct RealCylindricalGrid3d: public dg::aRealProductGeometry3d<real_type>
{
using perpendicular_grid = RealCartesianGrid2d<real_type>;
RealCylindricalGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}){}
RealCylindricalGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): dg::aRealProductGeometry3d<real_type>(gx,gy,gz){}
virtual RealCylindricalGrid3d* clone()const override final{
return new RealCylindricalGrid3d(*this);
}
private:
virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
return new RealCartesianGrid2d<real_type>(this->gx(), this->gy());
}
virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final{
SparseTensor<thrust::host_vector<real_type> > metric(*this);
thrust::host_vector<real_type> R = dg::evaluate(dg::cooX3d, *this);
for( unsigned i = 0; i<this->size(); i++)
R[i] = 1./R[i]/R[i];
metric.idx(2,2)=2;
metric.values().push_back( R);
return metric;
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny,unsigned Ny, unsigned nz,unsigned Nz) override final {
aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};


using aGeometry2d           = dg::aRealGeometry2d<double>;
using aGeometry3d           = dg::aRealGeometry3d<double>;
using aProductGeometry3d    = dg::aRealProductGeometry3d<double>;
using CartesianGrid2d       = dg::RealCartesianGrid2d<double>;
using CartesianGrid3d       = dg::RealCartesianGrid3d<double>;
using CylindricalGrid3d     = dg::RealCylindricalGrid3d<double>;
#ifndef MPI_VERSION
namespace x{
using aGeometry2d           = aGeometry2d           ;
using aGeometry3d           = aGeometry3d           ;
using aProductGeometry3d    = aProductGeometry3d    ;
using CartesianGrid2d       = CartesianGrid2d       ;
using CartesianGrid3d       = CartesianGrid3d       ;
using CylindricalGrid3d     = CylindricalGrid3d     ;
}
#endif 


} 
