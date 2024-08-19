#pragma once

#include "mpi_grid.h"
#include "base_geometry.h"
#include "tensor.h"

namespace dg
{



template<class real_type>
struct aRealMPIGeometry2d : public aRealMPITopology2d<real_type>
{
SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > jacobian()const {
return do_compute_jacobian();
}
SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > metric()const {
return do_compute_metric();
}
std::vector<MPI_Vector<thrust::host_vector<real_type>> > map()const{
return do_compute_map();
}
virtual aRealMPIGeometry2d* clone()const=0;
virtual aRealGeometry2d<real_type>* global_geometry()const =0;
virtual ~aRealMPIGeometry2d() = default;
protected:
using aRealMPITopology2d<real_type>::aRealMPITopology2d;
aRealMPIGeometry2d( const aRealMPIGeometry2d& src) = default;
aRealMPIGeometry2d& operator=( const aRealMPIGeometry2d& src) = default;
private:
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const {
return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_jacobian()const {
return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
}
virtual std::vector<MPI_Vector<thrust::host_vector<real_type>> > do_compute_map()const{
std::vector<MPI_Vector<thrust::host_vector<real_type>>> map(2);
map[0] = dg::evaluate(dg::cooX2d, *this);
map[1] = dg::evaluate(dg::cooY2d, *this);
return map;
}
};


template<class real_type>
struct aRealMPIGeometry3d : public aRealMPITopology3d<real_type>
{
SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > jacobian()const{
return do_compute_jacobian();
}
SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > metric()const {
return do_compute_metric();
}
std::vector<MPI_Vector<thrust::host_vector<real_type>> > map()const{
return do_compute_map();
}
virtual aRealMPIGeometry3d* clone()const=0;
virtual aRealGeometry3d<real_type>* global_geometry()const =0;
virtual ~aRealMPIGeometry3d() = default;
protected:
using aRealMPITopology3d<real_type>::aRealMPITopology3d;
aRealMPIGeometry3d( const aRealMPIGeometry3d& src) = default;
aRealMPIGeometry3d& operator=( const aRealMPIGeometry3d& src) = default;
private:
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const {
return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_jacobian()const {
return SparseTensor<MPI_Vector<thrust::host_vector<real_type>> >(*this);
}
virtual std::vector<MPI_Vector<thrust::host_vector<real_type>> > do_compute_map()const{
std::vector<MPI_Vector<thrust::host_vector<real_type>>> map(3);
map[0] = dg::evaluate(dg::cooX3d, *this);
map[1] = dg::evaluate(dg::cooY3d, *this);
map[2] = dg::evaluate(dg::cooZ3d, *this);
return map;
}
};

template<class real_type>
struct aRealProductMPIGeometry3d : public aRealMPIGeometry3d<real_type>
{

aRealMPIGeometry2d<real_type>* perp_grid()const{
return do_perp_grid();
}
virtual ~aRealProductMPIGeometry3d() = default;
virtual aRealProductMPIGeometry3d* clone()const=0;
protected:
using aRealMPIGeometry3d<real_type>::aRealMPIGeometry3d;
aRealProductMPIGeometry3d( const aRealProductMPIGeometry3d& src) = default;
aRealProductMPIGeometry3d& operator=( const aRealProductMPIGeometry3d& src) = default;
private:
virtual aRealMPIGeometry2d<real_type>* do_perp_grid()const=0;
};




template<class real_type>
struct RealCartesianMPIGrid2d : public aRealMPIGeometry2d<real_type>
{
RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): aRealMPIGeometry2d<real_type>( {x0, x1, n, Nx, dg::PER}, {y0, y1, n, Ny, dg::PER}, comm){}

RealCartesianMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):dg::aRealMPIGeometry2d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, comm){}
RealCartesianMPIGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, MPI_Comm comm): dg::aRealMPIGeometry2d<real_type>(gx,gy,comm){}
RealCartesianMPIGrid2d( const dg::RealMPIGrid2d<real_type>& g): aRealMPIGeometry2d<real_type>( g.global().gx(),g.global().gy(),g.communicator()){}
virtual RealCartesianMPIGrid2d* clone()const override final{return new RealCartesianMPIGrid2d(*this);}
virtual RealCartesianGrid2d<real_type>* global_geometry()const override final{
return new RealCartesianGrid2d<real_type>(
this->global().gx(), this->global().gy());
}
private:
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
aRealMPITopology2d<real_type>::do_set(nx,Nx,ny,Ny);
}

};


template<class real_type>
struct RealCartesianMPIGrid3d : public aRealProductMPIGeometry3d<real_type>
{
using perpendicular_grid = RealCartesianMPIGrid2d<real_type>;
RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): aRealProductMPIGeometry3d<real_type>( {x0, x1, n, Nx, dg::PER}, {y0, y1, n, Ny, dg::PER}, {z0, z1, 1, Nz, dg::PER}, comm){}

RealCartesianMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, {z0, z1, 1, Nz, bcz}, comm){}

RealCartesianMPIGrid3d( const dg::RealMPIGrid3d<real_type>& g): aRealProductMPIGeometry3d<real_type>( g.global().gx(), g.global().gy(), g.global().gz(),g.communicator()){}
virtual RealCartesianMPIGrid3d* clone()const override final{
return new RealCartesianMPIGrid3d(*this);
}
RealCartesianMPIGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): dg::aRealProductMPIGeometry3d<real_type>(gx,gy,gz,comm){}
virtual RealCartesianGrid3d<real_type>* global_geometry()const override final{
return new RealCartesianGrid3d<real_type>(
this->global().gx(), this->global().gy(), this->global().gz());
}

private:
virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
return new RealCartesianMPIGrid2d<real_type>( this->global().gx(), this->global().gy(), this->get_perp_comm( ));
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny,unsigned Ny, unsigned nz,unsigned Nz)override final{
aRealMPITopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};


template<class real_type>
struct RealCylindricalMPIGrid3d: public aRealProductMPIGeometry3d<real_type>
{
using perpendicular_grid = RealCartesianMPIGrid2d<real_type>;
RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, {z0, z1, 1, Nz, bcz}, comm){}
RealCylindricalMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, MPI_Comm comm):aRealProductMPIGeometry3d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, {z0, z1, 1, Nz, dg::PER}, comm){}

RealCylindricalMPIGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): dg::aRealProductMPIGeometry3d<real_type>(gx,gy,gz,comm){}

virtual RealCylindricalMPIGrid3d<real_type>* clone()const override final{
return new RealCylindricalMPIGrid3d(*this);
}
virtual RealCylindricalGrid3d<real_type>* global_geometry()const override final{
return new RealCylindricalGrid3d<real_type>(
this->global().gx(), this->global().gy(), this->global().gz());
}
private:
virtual RealCartesianMPIGrid2d<real_type>* do_perp_grid()const override final{
return new RealCartesianMPIGrid2d<real_type>( this->global().gx(), this->global().gy(), this->get_perp_comm( ));
}
virtual SparseTensor<MPI_Vector<thrust::host_vector<real_type>> > do_compute_metric()const override final{
SparseTensor<MPI_Vector<thrust::host_vector<real_type>>> metric(*this);
MPI_Vector<thrust::host_vector<real_type>> R = dg::evaluate(dg::cooX3d, *this);
for( unsigned i = 0; i<this->local().size(); i++)
R.data()[i] = 1./R.data()[i]/R.data()[i];
metric.idx(2,2)=2;
metric.values().push_back(R);
return metric;
}
virtual void do_set(unsigned nx, unsigned Nx, unsigned ny,unsigned Ny, unsigned nz,unsigned Nz) override final{
aRealMPITopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};

using aMPIGeometry2d        = dg::aRealMPIGeometry2d<double>;
using aMPIGeometry3d        = dg::aRealMPIGeometry3d<double>;
using aProductMPIGeometry3d = dg::aRealProductMPIGeometry3d<double>;
using CartesianMPIGrid2d    = dg::RealCartesianMPIGrid2d<double>;
using CartesianMPIGrid3d    = dg::RealCartesianMPIGrid3d<double>;
using CylindricalMPIGrid3d  = dg::RealCylindricalMPIGrid3d<double>;
namespace x{
using aGeometry2d           = aMPIGeometry2d           ;
using aGeometry3d           = aMPIGeometry3d           ;
using aProductGeometry3d    = aProductMPIGeometry3d    ;
using CartesianGrid2d       = CartesianMPIGrid2d       ;
using CartesianGrid3d       = CartesianMPIGrid3d       ;
using CylindricalGrid3d     = CylindricalMPIGrid3d     ;
}

}
