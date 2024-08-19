#pragma once
#include <cassert>
#include <cmath> 
#include "dlt.h"
#include "grid.h"
#include "../enums.h"







namespace dg{
template<class real_type>
struct RealGridX2d;
template<class real_type>
struct RealGridX3d;


template<class real_type>
struct RealGridX1d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGridX1d<real_type>;

RealGridX1d( real_type x0, real_type x1, real_type f, unsigned n, unsigned N, bc bcx = NEU):
x0_(x0), x1_(x1), f_(f),
n_(n), Nx_(N), bcx_(bcx), dlt_(n)
{
assert( (f >= 0) && (f < 0.5) );
assert( fabs(outer_N() - f*(real_type)N) < 1e-14);
assert( x1 > x0 );
assert( N > 0  );
assert( n != 0 );
assert( bcx != PER);
}

real_type x0() const {return x0_;}

real_type x1() const {return x1_;}

real_type f() const {return f_;}

real_type lx() const {return x1_-x0_;}

real_type h() const {return lx()/(real_type)Nx_;}

unsigned N() const {return Nx_;}

unsigned outer_N() const {return (unsigned)(round(f_*(real_type)Nx_));}

unsigned inner_N() const {return N()-2*outer_N();}

unsigned n() const {return n_;}

bc bcx() const {return bcx_;}

unsigned size() const { return n_*Nx_;}

void display( std::ostream& os = std::cout) const
{
os << "RealGrid parameters are: \n"
<<"    n  = "<<n_<<"\n"
<<"    N  = "<<Nx_<<"\n"
<<"    inner N = "<<inner_N()<<"\n"
<<"    outer N = "<<outer_N()<<"\n"
<<"    h  = "<<h()<<"\n"
<<"    x0 = "<<x0_<<"\n"
<<"    x1 = "<<x1_<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx_)<<"\n";
}

const DLT<real_type>& dlt() const {return dlt_;}
RealGrid1d<real_type> grid() const{return RealGrid1d<real_type>( x0_, x1_, n_, Nx_, bcx_);}


void shift_topologic( real_type x0, real_type& x1) const
{
assert( contains(x0));
real_type deltaX;
real_type xleft = x0_ + f_*lx();
real_type xright = x1_ - f_*lx();
if( x0 >= xleft && x0<xright)
{
if( x1 > xleft) deltaX = (x1 -xleft);
else deltaX = xright - x1;
unsigned N = floor(deltaX/(xright-xleft));
if( x1  > xright ) x1 -= N*lx();
if( x1  < xleft ) x1 += N*lx();
}
else if( x0 < xleft && x1 >=xleft)
x1 += (xright-xleft);
else if( x0 >= xright  && x1 < xright)
x1 -= (xright-xleft);

}


bool contains( real_type x) const
{
if( (x>=x0_ && x <= x1_)) return true;
return false;
}
private:
real_type x0_, x1_, f_;
unsigned n_, Nx_;
bc bcx_;
DLT<real_type> dlt_;
};



template<class real_type>
struct aRealTopologyX2d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGridX2d<real_type>;


real_type x0() const {return x0_;}

real_type x1() const {return x1_;}

real_type y0() const {return y0_;}

real_type y1() const {return y1_;}

real_type lx() const {return x1_-x0_;}

real_type ly() const {return y1_-y0_;}

real_type hx() const {return lx()/(real_type)Nx_;}

real_type hy() const {return ly()/(real_type)Ny_;}

real_type fx() const {return fx_;}

real_type fy() const {return fy_;}

unsigned n() const {return n_;}

unsigned Nx() const {return Nx_;}

unsigned inner_Nx() const {return Nx_ - outer_Nx();}

unsigned outer_Nx() const {return (unsigned)round(fx_*(real_type)Nx_);}

unsigned Ny() const {return Ny_;}

unsigned inner_Ny() const {return Ny_-2*outer_Ny();}

unsigned outer_Ny() const {return (unsigned)round(fy_*(real_type)Ny_);}

bc bcx() const {return bcx_;}

bc bcy() const {return bcy_;}

RealGrid2d<real_type> grid() const {return RealGrid2d<real_type>( x0_,x1_,y0_,y1_,n_,Nx_,Ny_,bcx_,bcy_);}

const DLT<real_type>& dlt() const{return dlt_;}

unsigned size() const { return n_*n_*Nx_*Ny_;}

void display( std::ostream& os = std::cout) const
{
os << "Grid parameters are: \n"
<<"    n  = "<<n_<<"\n"
<<"    Nx = "<<Nx_<<"\n"
<<"    inner Nx = "<<inner_Nx()<<"\n"
<<"    outer Nx = "<<outer_Nx()<<"\n"
<<"    Ny = "<<Ny_<<"\n"
<<"    inner Ny = "<<inner_Ny()<<"\n"
<<"    outer Ny = "<<outer_Ny()<<"\n"
<<"    hx = "<<hx()<<"\n"
<<"    hy = "<<hy()<<"\n"
<<"    x0 = "<<x0_<<"\n"
<<"    x1 = "<<x1_<<"\n"
<<"    y0 = "<<y0_<<"\n"
<<"    y1 = "<<y1_<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"    ly = "<<ly()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx_)<<"\n"
<<"Boundary conditions in y are: \n"
<<"    "<<bc2str(bcy_)<<"\n";
}


void shift_topologic( real_type x0, real_type y0, real_type& x1, real_type& y1) const
{
assert( contains(x0, y0));
real_type deltaX;
if( x1 > x0_) deltaX = (x1 -x0_);
else deltaX = x1_ - x1;
unsigned N = floor(deltaX/lx());
if( x1  > x1_ && bcx_ == dg::PER) x1 -= N*lx();
if( x1  < x0_ && bcx_ == dg::PER) x1 += N*lx();

if( x0 < x1_ - fx_*(x1_-x0_) ) 
{
real_type deltaY;
real_type yleft = y0_ + fy_*ly();
real_type yright = y1_ - fy_*ly();
if( y0 >= yleft && y0<yright)
{
if( y1 > yleft) deltaY = (y1 -yleft);
else deltaY = yright - y1;
unsigned N = floor(deltaY/(yright-yleft));
if( y1  > yright ) y1 -= N*ly();
if( y1  < yleft ) y1 += N*ly();
}
else if( y0 < yleft && y1 >=yleft)
y1 += (yright-yleft);
else if( y0 >= yright  && y1 < yright)
y1 -= (yright-yleft);
}

}


bool contains( real_type x, real_type y)const
{
if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_)) return true;
return false;
}
protected:
~aRealTopologyX2d() = default;
aRealTopologyX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):
x0_(x0), x1_(x1), y0_(y0), y1_(y1), fx_(fx), fy_(fy),
n_(n), Nx_(Nx), Ny_(Ny), bcx_(bcx), bcy_( bcy), dlt_(n)
{
assert( (fy_ >= 0.) && (fy_ < 0.5) );
assert( (fx_ >= 0.) && (fx_ < 1.) );
assert( fabs(outer_Nx() - fx_*(real_type)Nx) < 1e-14);
assert( fabs(outer_Ny() - fy_*(real_type)Ny) < 1e-14);
assert( n != 0);
assert( x1 > x0 && y1 > y0);
assert( Nx_ > 0  && Ny > 0 );
assert( bcy != PER);
}
aRealTopologyX2d(const aRealTopologyX2d& src) = default;
aRealTopologyX2d& operator=(const aRealTopologyX2d& src) = default;
private:
real_type x0_, x1_, y0_, y1_;
real_type fx_, fy_;
unsigned n_, Nx_, Ny_;
bc bcx_, bcy_;
DLT<real_type> dlt_;
};

template<class real_type>
struct RealGridX2d : public aRealTopologyX2d<real_type>
{
RealGridX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx=PER, bc bcy=NEU):
aRealTopologyX2d<real_type>(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy) { }
explicit RealGridX2d( const aRealTopologyX2d<real_type>& src): aRealTopologyX2d<real_type>(src){}
};


template<class real_type>
struct aRealTopologyX3d
{
using value_type = real_type;
using host_vector = thrust::host_vector<real_type>;
using host_grid = RealGridX3d<real_type>;

real_type x0() const {return x0_;}

real_type x1() const {return x1_;}


real_type y0() const {return y0_;}

real_type y1() const {return y1_;}


real_type z0() const {return z0_;}

real_type z1() const {return z1_;}


real_type lx() const {return x1_-x0_;}

real_type ly() const {return y1_-y0_;}

real_type lz() const {return z1_-z0_;}


real_type hx() const {return lx()/(real_type)Nx_;}

real_type hy() const {return ly()/(real_type)Ny_;}

real_type hz() const {return lz()/(real_type)Nz_;}

real_type fx() const {return fx_;}

real_type fy() const {return fy_;}

unsigned n() const {return n_;}

unsigned Nx() const {return Nx_;}

unsigned inner_Nx() const {return Nx_ - outer_Nx();}

unsigned outer_Nx() const {return (unsigned)round(fx_*(real_type)Nx_);}

unsigned Ny() const {return Ny_;}

unsigned inner_Ny() const {return Ny_-2*outer_Ny();}

unsigned outer_Ny() const {return (unsigned)round(fy_*(real_type)Ny_);}

unsigned Nz() const {return Nz_;}

bc bcx() const {return bcx_;}

bc bcy() const {return bcy_;}

bc bcz() const {return bcz_;}

RealGrid3d<real_type> grid() const {
return RealGrid3d<real_type>( x0_,x1_,y0_,y1_,z0_,z1_,n_,Nx_,Ny_,Nz_,bcx_,bcy_,bcz_);
}

const DLT<real_type>& dlt() const{return dlt_;}

unsigned size() const { return n_*n_*Nx_*Ny_*Nz_;}

void display( std::ostream& os = std::cout) const
{
os << "Grid parameters are: \n"
<<"    n  = "<<n_<<"\n"
<<"    Nx = "<<Nx_<<"\n"
<<"    inner Nx = "<<inner_Nx()<<"\n"
<<"    outer Nx = "<<outer_Nx()<<"\n"
<<"    Ny = "<<Ny_<<"\n"
<<"    inner Ny = "<<inner_Ny()<<"\n"
<<"    outer Ny = "<<outer_Ny()<<"\n"
<<"    Nz = "<<Nz_<<"\n"
<<"    hx = "<<hx()<<"\n"
<<"    hy = "<<hy()<<"\n"
<<"    hz = "<<hz()<<"\n"
<<"    x0 = "<<x0_<<"\n"
<<"    x1 = "<<x1_<<"\n"
<<"    y0 = "<<y0_<<"\n"
<<"    y1 = "<<y1_<<"\n"
<<"    z0 = "<<z0_<<"\n"
<<"    z1 = "<<z1_<<"\n"
<<"    lx = "<<lx()<<"\n"
<<"    ly = "<<ly()<<"\n"
<<"    lz = "<<lz()<<"\n"
<<"Boundary conditions in x are: \n"
<<"    "<<bc2str(bcx_)<<"\n"
<<"Boundary conditions in y are: \n"
<<"    "<<bc2str(bcy_)<<"\n"
<<"Boundary conditions in z are: \n"
<<"    "<<bc2str(bcz_)<<"\n";
}

bool contains( real_type x, real_type y, real_type z)const
{
if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_) && (z>=z0_ && z<=z1_))
return true;
return false;
}
protected:
~aRealTopologyX3d() = default;
aRealTopologyX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz):
x0_(x0), x1_(x1), y0_(y0), y1_(y1), z0_(z0), z1_(z1), fx_(fx), fy_(fy),
n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), bcx_(bcx), bcy_( bcy), bcz_( bcz), dlt_(n)
{
assert( (fy_ >= 0.) && (fy_ < 0.5) );
assert( (fx_ >= 0.) && (fx_ < 1.) );
assert( fabs(outer_Nx() - fx_*(real_type)Nx) < 1e-14);
assert( fabs(outer_Ny() - fy_*(real_type)Ny) < 1e-14);
assert( n != 0);
assert( x1 > x0 && y1 > y0 ); assert( z1 > z0 );
assert( Nx_ > 0  && Ny > 0); assert( Nz > 0);
}
aRealTopologyX3d(const aRealTopologyX3d& src) = default;
aRealTopologyX3d& operator=(const aRealTopologyX3d& src) = default;
private:
real_type x0_, x1_, y0_, y1_, z0_, z1_;
real_type fx_,fy_;
unsigned n_, Nx_, Ny_, Nz_;
bc bcx_, bcy_, bcz_;
DLT<real_type> dlt_;
};


template<class real_type>
struct RealGridX3d : public aRealTopologyX3d<real_type>
{
RealGridX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=PER, bc bcy=NEU, bc bcz=PER):
aRealTopologyX3d<real_type>(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz) { }
explicit RealGridX3d( const aRealTopologyX3d<real_type>& src): aRealTopologyX3d<real_type>(src){}
};

using GridX1d       = dg::RealGridX1d<double>;
using GridX2d       = dg::RealGridX2d<double>;
using GridX3d       = dg::RealGridX3d<double>;
using aTopologyX2d  = dg::aRealTopologyX2d<double>;
using aTopologyX3d  = dg::aRealTopologyX3d<double>;

}
