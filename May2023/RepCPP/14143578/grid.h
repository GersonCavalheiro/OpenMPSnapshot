#pragma once

#include <cassert>
#include <cmath>
#include <thrust/host_vector.h>
#include "topological_traits.h"
#include "dlt.h"
#include "../enums.h"









namespace dg{

template<class real_type>
struct RealGrid2d;
template<class real_type>
struct RealGrid3d;


template<class real_type>
struct RealGrid1d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGrid1d<real_type>;

RealGrid1d() = default;

RealGrid1d( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx = PER)
{
set(x0,x1,bcx);
set(n,N);
}

real_type x0() const {return x0_;}

real_type x1() const {return x1_;}

real_type lx() const {return x1_-x0_;}

real_type h() const {return lx()/(real_type)Nx_;}

unsigned N() const {return Nx_;}

unsigned n() const {return n_;}

bc bcx() const {return bcx_;}

void set(real_type x0, real_type x1, bc bcx)
{
assert( x1 > x0 );
x0_=x0, x1_=x1;
bcx_=bcx;
}

void set( unsigned n, unsigned N)
{
assert( N > 0  );
Nx_=N; n_=n;
dlt_=DLT<real_type>(n);
}

void set( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx)
{
set(x0,x1,bcx);
set(n,N);
}

unsigned size() const { return n_*Nx_;}

const DLT<real_type>& dlt() const {return dlt_;}
void display( std::ostream& os = std::cout) const
{
os << "Topology parameters are: \n"
<<"    n  = "<<n_<<"\n"
<<"    N = "<<Nx_<<"\n"
<<"    h = "<<h()<<"\n"
<<"    x0 = "<<x0_<<"\n"
<<"    x1 = "<<x1_<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx_)<<"\n";
}


void shift( bool& negative, real_type& x)const
{
shift( negative, x, bcx_);
}

void shift( bool& negative, real_type &x, bc bcx)const
{
if( bcx == dg::PER)
{
real_type N = floor((x-x0_)/(x1_-x0_)); 
x = x - N*(x1_-x0_); 
}
while( (x<x0_) || (x>x1_) )
{
if( x < x0_){
x = 2.*x0_ - x;
if( bcx == dg::DIR || bcx == dg::DIR_NEU)
negative = !negative;
}
if( x > x1_){
x = 2.*x1_ - x;
if( bcx == dg::DIR || bcx == dg::NEU_DIR) 
negative = !negative; 
}
}
}


bool contains( real_type x)const
{
if( !std::isfinite(x) ) return false;
if( (x>=x0_ && x <= x1_)) return true;
return false;
}

private:
real_type x0_, x1_;
unsigned n_, Nx_;
bc bcx_;
DLT<real_type> dlt_;
};


template<class real_type>
struct aRealTopology2d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGrid2d<real_type>;


real_type x0() const {return gx_.x0();}

real_type x1() const {return gx_.x1();}

real_type y0() const {return gy_.x0();}

real_type y1() const {return gy_.x1();}

real_type lx() const {return gx_.lx();}

real_type ly() const {return gy_.lx();}

real_type hx() const {return gx_.h();}

real_type hy() const {return gy_.h();}

unsigned n() const {return gx_.n();}
unsigned nx() const {return gx_.n();}
unsigned ny() const {return gy_.n();}

unsigned Nx() const {return gx_.N();}

unsigned Ny() const {return gy_.N();}

bc bcx() const {return gx_.bcx();}

bc bcy() const {return gy_.bcx();}

const DLT<real_type>& dltx() const{return gx_.dlt();}
const DLT<real_type>& dlty() const{return gy_.dlt();}

const RealGrid1d<real_type>& gx() const {return gx_;}
const RealGrid1d<real_type>& gy() const {return gy_;}


void multiplyCellNumbers( real_type fx, real_type fy){
if( fx != 1 || fy != 1)
do_set(nx(), round(fx*(real_type)Nx()), ny(), round(fy*(real_type)Ny()));
}

void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) {
set( new_n, new_Nx, new_n, new_Ny);
}

void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny) {
if( new_nx==nx() && new_Nx==Nx() && new_ny==ny() && new_Ny == Ny())
return;
do_set(new_nx,new_Nx,new_ny,new_Ny);
}



unsigned size() const { return gx_.size()*gy_.size();}

void display( std::ostream& os = std::cout) const
{
os << "Topology parameters are: \n"
<<"    nx = "<<nx()<<"\n"
<<"    ny = "<<ny()<<"\n"
<<"    Nx = "<<Nx()<<"\n"
<<"    Ny = "<<Ny()<<"\n"
<<"    hx = "<<hx()<<"\n"
<<"    hy = "<<hy()<<"\n"
<<"    x0 = "<<x0()<<"\n"
<<"    x1 = "<<x1()<<"\n"
<<"    y0 = "<<y0()<<"\n"
<<"    y1 = "<<y1()<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"    ly = "<<ly()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx())<<"\n"
<<"Boundary conditions in y are: \n"
<<"    "<<bc2str(bcy())<<"\n";
}

void shift( bool& negative, real_type& x, real_type& y)const
{
shift( negative, x, y, bcx(), bcy());
}

void shift( bool& negative, real_type& x, real_type& y, bc bcx, bc bcy)const
{
gx_.shift( negative, x,bcx);
gy_.shift( negative, y,bcy);
}

bool contains( real_type x, real_type y)const
{
if( gx_.contains(x) && gy_.contains(y)) return true;
return false;
}
template<class Vector>
bool contains( const Vector& x) const{
return contains( x[0], x[1]);
}
protected:
~aRealTopology2d() = default;

aRealTopology2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): gx_(gx),gy_(gy) { }

aRealTopology2d(const aRealTopology2d& src) = default;
aRealTopology2d& operator=(const aRealTopology2d& src) = default;
virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny,
unsigned new_Ny)=0;
private:
RealGrid1d<real_type> gx_, gy_;
};




template<class real_type>
struct aRealTopology3d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGrid3d<real_type>;


real_type x0() const {return gx_.x0();}

real_type x1() const {return gx_.x1();}


real_type y0() const {return gy_.x0();}

real_type y1() const {return gy_.x1();}


real_type z0() const {return gz_.x0();}

real_type z1() const {return gz_.x1();}


real_type lx() const {return gx_.lx();}

real_type ly() const {return gy_.lx();}

real_type lz() const {return gz_.lx();}


real_type hx() const {return gx_.h();}

real_type hy() const {return gy_.h();}

real_type hz() const {return gz_.h();}

unsigned n() const {return gx_.n();}
unsigned nx() const {return gx_.n();}
unsigned ny() const {return gy_.n();}
unsigned nz() const {return gz_.n();}

unsigned Nx() const {return gx_.N();}

unsigned Ny() const {return gy_.N();}

unsigned Nz() const {return gz_.N();}

bc bcx() const {return gx_.bcx();}

bc bcy() const {return gy_.bcx();}

bc bcz() const {return gz_.bcx();}
const DLT<real_type>& dltx() const{return gx_.dlt();}
const DLT<real_type>& dlty() const{return gy_.dlt();}
const DLT<real_type>& dltz() const{return gz_.dlt();}
const RealGrid1d<real_type>& gx() const {return gx_;}
const RealGrid1d<real_type>& gy() const {return gy_;}
const RealGrid1d<real_type>& gz() const {return gz_;}

unsigned size() const { return gx_.size()*gy_.size()*gz_.size();}

void display( std::ostream& os = std::cout) const
{
os << "Topology parameters are: \n"
<<"    nx = "<<nx()<<"\n"
<<"    ny = "<<ny()<<"\n"
<<"    nz = "<<nz()<<"\n"
<<"    Nx = "<<Nx()<<"\n"
<<"    Ny = "<<Ny()<<"\n"
<<"    Nz = "<<Nz()<<"\n"
<<"    hx = "<<hx()<<"\n"
<<"    hy = "<<hy()<<"\n"
<<"    hz = "<<hz()<<"\n"
<<"    x0 = "<<x0()<<"\n"
<<"    x1 = "<<x1()<<"\n"
<<"    y0 = "<<y0()<<"\n"
<<"    y1 = "<<y1()<<"\n"
<<"    z0 = "<<z0()<<"\n"
<<"    z1 = "<<z1()<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"    ly = "<<ly()<<"\n"
<<"    lz = "<<lz()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx())<<"\n"
<<"Boundary conditions in y are: \n"
<<"    "<<bc2str(bcy())<<"\n"
<<"Boundary conditions in z are: \n"
<<"    "<<bc2str(bcz())<<"\n";
}


void shift( bool& negative, real_type& x, real_type& y, real_type& z)const
{
shift( negative, x,y,z, bcx(), bcy(), bcz());
}

void shift( bool& negative, real_type& x, real_type& y, real_type& z, bc bcx, bc bcy, bc bcz)const
{
gx_.shift( negative, x,bcx);
gy_.shift( negative, y,bcy);
gz_.shift( negative, z,bcz);
}


bool contains( real_type x, real_type y, real_type z)const
{
if( gx_.contains(x) && gy_.contains(y) && gz_.contains(z))
return true;
return false;
}
template<class Vector>
bool contains( const Vector& x) const{
return contains( x[0], x[1], x[2]);
}
void multiplyCellNumbers( real_type fx, real_type fy){
if( fx != 1 || fy != 1)
do_set(nx(), round(fx*(real_type)Nx()), ny(),
round(fy*(real_type)Ny()), nz(), Nz());
}

void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) {
set(new_n,new_Nx,new_n,new_Ny,1,new_Nz);
}

void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz) {
if( new_nx==nx() && new_Nx ==Nx() && new_ny == ny() && new_Ny == Ny() && new_nz == nz() && new_Nz==Nz())
return;
do_set(new_nx,new_Nx,new_ny,new_Ny,new_nz,new_Nz);
}
protected:
~aRealTopology3d() = default;

aRealTopology3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz):
gx_(gx),gy_(gy),gz_(gz){
}
aRealTopology3d(const aRealTopology3d& src) = default;
aRealTopology3d& operator=(const aRealTopology3d& src) = default;
virtual void do_set(unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)=0;
private:
RealGrid1d<real_type> gx_,gy_,gz_;
};


template<class real_type>
struct RealGrid2d : public aRealTopology2d<real_type>
{
RealGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER):
aRealTopology2d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy}) { }

RealGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): aRealTopology2d<real_type>(gx,gy){ }

explicit RealGrid2d( const aRealTopology2d<real_type>& src): aRealTopology2d<real_type>(src){}
private:
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
aRealTopology2d<real_type>::do_set(nx,Nx,ny,Ny);
}

};


template<class real_type>
struct RealGrid3d : public aRealTopology3d<real_type>
{
RealGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz=PER):
aRealTopology3d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}) { }
RealGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): aRealTopology3d<real_type>(gx,gy,gz){ }

explicit RealGrid3d( const aRealTopology3d<real_type>& src): aRealTopology3d<real_type>(src){ }
private:
virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny,
unsigned nz, unsigned Nz) override final{
aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
}
};

template<class real_type>
void aRealTopology2d<real_type>::do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny)
{
gx_.set(new_nx, new_Nx);
gy_.set(new_ny, new_Ny);
}
template<class real_type>
void aRealTopology3d<real_type>::do_set(unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)
{
gx_.set(new_nx, new_Nx);
gy_.set(new_ny, new_Ny);
gz_.set(new_nz, new_Nz);
}

template<class Topology>
using get_host_vector = typename Topology::host_vector;

template<class Topology>
using get_host_grid = typename Topology::host_grid;


using Grid1d        = dg::RealGrid1d<double>;
using Grid2d        = dg::RealGrid2d<double>;
using Grid3d        = dg::RealGrid3d<double>;
using aTopology2d   = dg::aRealTopology2d<double>;
using aTopology3d   = dg::aRealTopology3d<double>;
#ifndef MPI_VERSION
namespace x {
using Grid1d        = Grid1d      ;
using Grid2d        = Grid2d      ;
using Grid3d        = Grid3d      ;
using aTopology2d   = aTopology2d ;
using aTopology3d   = aTopology3d ;
} 
#endif

}
