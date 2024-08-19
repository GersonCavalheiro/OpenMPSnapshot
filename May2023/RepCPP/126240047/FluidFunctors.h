#ifndef FLUID_FUNCTORS_H_
#define FLUID_FUNCTORS_H_

#include "kokkos_shared.h"

#include <limits> 
#ifdef __CUDA_ARCH__
#include <math_constants.h> 
#endif 

#include "FluidQuantity.h"


class AdvectionFunctor
{

public:


AdvectionFunctor(FluidQuantity data,
FluidQuantity u,
FluidQuantity v,
double timestep) :
data(data),
u(u),
v(v),
timestep(timestep),
_w(data._w),
_h(data._h),
_ox(data._ox),
_oy(data._oy),
_hx(data._hx)
{
};

static void apply(FluidQuantity data,
FluidQuantity u,
FluidQuantity v,
double timestep) {
const int size = data._w*data._h;
AdvectionFunctor functor(data, u, v, timestep);
Kokkos::parallel_for(size, functor);
}

KOKKOS_INLINE_FUNCTION
void euler(double &x,
double &y) const {

double uVel = u.lerp(x, y)/_hx;
double vVel = v.lerp(x, y)/_hx;

x -= uVel*timestep;
y -= vVel*timestep;

} 


KOKKOS_INLINE_FUNCTION
void rungeKutta3(double &x,
double &y) const {

double firstU = u.lerp(x, y)/_hx;
double firstV = v.lerp(x, y)/_hx;

double midX = x - 0.5*timestep*firstU;
double midY = y - 0.5*timestep*firstV;

double midU = u.lerp(midX, midY)/_hx;
double midV = v.lerp(midX, midY)/_hx;

double lastX = x - 0.75*timestep*midU;
double lastY = y - 0.75*timestep*midV;

double lastU = u.lerp(lastX, lastY);
double lastV = v.lerp(lastX, lastY);

x -= timestep*((2.0/9.0)*firstU + (3.0/9.0)*midU + (4.0/9.0)*lastU);
y -= timestep*((2.0/9.0)*firstV + (3.0/9.0)*midV + (4.0/9.0)*lastV);

} 


KOKKOS_INLINE_FUNCTION
void operator() (const int& index) const
{

int ix, iy;
index2coord(index,ix,iy,_w,_h);

double x = ix + _ox;
double y = iy + _oy;

rungeKutta3(x, y);

data._dst(ix,iy) = data.cerp(x, y);

} 

FluidQuantity data;
FluidQuantity u;
FluidQuantity v;
double timestep;
int _w;
int _h;
double _ox;
double _oy;
double _hx;

}; 


class InflowFunctor
{

public:


InflowFunctor(FluidQuantity& fq, double v, double x0, double y0, double x1, double y1) :
x0(x0),
y0(y0),
x1(x1),
y1(y1),
_w(fq._w),
_h(fq._h),
_ox(fq._ox),
_oy(fq._oy),
_hx(fq._hx),
data( fq.src() ),
v(v)
{
ix0 = (int)(x0/_hx - _ox); ix0 = ix0>0  ? ix0 : 0;
ix1 = (int)(x1/_hx - _ox); ix1 = ix1<_w ? ix1 : _w;

iy0 = (int)(y0/_hx - _oy); iy0 = iy0>0  ? iy0 : 0;
iy1 = (int)(y1/_hx - _oy); iy1 = iy1<_h ? iy1 : _h;

};

static void apply(FluidQuantity& fq,
double v,
double x0, double y0, double x1, double y1)
{
const int size = fq._w*fq._h;
InflowFunctor functor(fq, v, x0, y0, x1, y1);
Kokkos::parallel_for(size, functor);
}

KOKKOS_INLINE_FUNCTION
void operator() (const int& index) const
{


int ix, iy;
index2coord(index,ix,iy,_w,_h);

if (ix >= ix0 and ix<ix1 and
iy >= iy0 and iy<iy1 ) {
double l = length(
(2.0*(ix + 0.5)*_hx - (x0 + x1))/(x1 - x0),
(2.0*(iy + 0.5)*_hx - (y0 + y1))/(y1 - y0)
);
double vi = cubicPulse(l)*v;
if (fabs(data(ix,iy)) < fabs(vi))
data(ix,iy) = vi;

}

} 

double x0, y0, x1, y1;
int _w, _h;
double _ox, _oy;
double _hx;
Array2d data;
double v;
int ix0, ix1;
int iy0, iy1;

}; 


class BuildRHSFunctor
{

public:


BuildRHSFunctor(Array2d r, Array2d u, Array2d v, double scale, int w, int h) :
_r(r),
_u(u),
_v(v),
scale(scale),
_w(w),
_h(h)
{};

static void apply(Array2d r, Array2d u, Array2d v, double scale, int w, int h)
{
const int size = w*h;
BuildRHSFunctor functor(r, u, v, scale, w, h);
Kokkos::parallel_for(size, functor);
}

KOKKOS_INLINE_FUNCTION
void operator() (const int& index) const
{

int x, y;
index2coord(index,x,y,_w,_h);

_r(x,y) = -scale * (_u(x + 1, y    ) - _u(x, y) +
_v(x    , y + 1) - _v(x, y) );

} 

Array2d _r;
Array2d _u, _v;
double scale;
int _w,_h;

}; 


class ApplyPressureFunctor
{

public:


ApplyPressureFunctor(Array2d p,
Array2d u, Array2d v,
double scale, int w, int h) :
_p(p),
_u(u),
_v(v),
_scale(scale),
_w(w),
_h(h)
{};

static void apply(Array2d p,
Array2d u, Array2d v,
double scale, int w, int h)
{
const int size = w*h;
ApplyPressureFunctor functor(p, u, v, scale, w, h);
Kokkos::parallel_for(size, functor);
}

KOKKOS_INLINE_FUNCTION
void operator() (const int& index) const
{


int x, y;
index2coord(index,x,y,_w,_h);

_u(x, y) -= _scale * _p(x  ,y  );
if (x>0)
_u(x, y) += _scale * _p(x-1,y  );

_v(x, y) -= _scale * _p(x  ,y  );
if (y>0)
_v(x, y) += _scale * _p(x  ,y-1);

if (x==0) {
_u(0 , y ) = 0.0;
_u(_w, y ) = 0.0;
}

if (y==0) {
_v(x , 0 ) = 0.0;
_v(x , _h) = 0.0;
}

} 

Array2d _p;
Array2d _u, _v;
double _scale;
int _w,_h;

}; 


class MaxVelocityFunctor
{

public:


MaxVelocityFunctor(FluidQuantity u, FluidQuantity v, int w, int h) :
_u(u),
_v(v),
_w(w),
_h(h)
{};

static void apply(FluidQuantity u, FluidQuantity v, double& maxVelocity,
int w, int h)
{
const int size = w*h;
MaxVelocityFunctor functor(u, v, w, h);
Kokkos::parallel_reduce(size, functor, maxVelocity);
}

KOKKOS_INLINE_FUNCTION
void init (double& dst) const
{
#ifdef __CUDA_ARCH__
dst = -CUDART_INF;
#else
dst = std::numeric_limits<double>::min();
#endif 
} 

KOKKOS_INLINE_FUNCTION
void operator() (const int& index, double& maxVelocity) const
{

int x, y;
index2coord(index,x,y,_w,_h);


double u = _u.lerp(x + 0.5, y + 0.5);
double v = _v.lerp(x + 0.5, y + 0.5);

double velocity = sqrt(u*u + v*v);
maxVelocity = fmax(maxVelocity, velocity);

} 

KOKKOS_INLINE_FUNCTION
void join (volatile double& dst,
const volatile double& src) const
{
if (dst < src) {
dst = src;
}
} 

FluidQuantity _u, _v;
int _w,_h;

}; 


class ProjectFunctor_GaussSeidel
{

public:

enum RedBlack_t {
RED = 0,
BLACK = 1
};


ProjectFunctor_GaussSeidel(Array2d p, Array2d r, double scale, int w, int h,
RedBlack_t redblack_type) :
_p(p),
_r(r),
_scale(scale),
_w(w),
_h(h),
redblack_type(redblack_type)
{};

static void apply(Array2d p, Array2d r, double scale,
double& maxDelta,
int w, int h,
RedBlack_t redblack_type)
{
const int size = w*h;
ProjectFunctor_GaussSeidel functor(p, r, scale, w, h, redblack_type);
Kokkos::parallel_reduce(size, functor, maxDelta);
}

KOKKOS_INLINE_FUNCTION
void init (double& dst) const
{
#ifdef __CUDA_ARCH__
dst = -CUDART_INF;
#else
dst = std::numeric_limits<double>::min();
#endif 
} 

KOKKOS_INLINE_FUNCTION
void do_red_black(int &x, int &y,
double &maxDelta) const
{

double diag = 0.0, offDiag = 0.0;


if (x > 0) {
diag    += _scale;
offDiag -= _scale*_p(x - 1, y    );
}
if (y > 0) {
diag    += _scale;
offDiag -= _scale*_p(x    , y - 1);
}
if (x < _w - 1) {
diag    += _scale;
offDiag -= _scale*_p(x + 1, y    );
}
if (y < _h - 1) {
diag    += _scale;
offDiag -= _scale*_p(x    , y + 1);
}

double newP = ( _r(x,y) - offDiag ) / diag;
maxDelta = fmax(maxDelta, fabs(_p(x,y) - newP));

_p(x,y) = newP;

} 


KOKKOS_INLINE_FUNCTION
void operator() (const int& index, double& maxDelta) const
{

int x, y;
index2coord(index,x,y,_w,_h);


if ( !((x+y+redblack_type)&1) ) {

do_red_black(x,y, maxDelta);

}

} 

KOKKOS_INLINE_FUNCTION
void join (volatile double& dst,
const volatile double& src) const
{
if (dst < src) {
dst = src;
}
} 

Array2d _p, _r;
double  _scale;
int     _w,_h;
RedBlack_t redblack_type;

}; 


class ProjectFunctor_Jacobi
{

public:


ProjectFunctor_Jacobi(Array2d p, Array2d p2, Array2d r, double scale, int w, int h) :
_p(p),
_p2(p2),
_r(r),
_scale(scale),
_w(w),
_h(h)
{};

static void apply(Array2d p, Array2d p2, Array2d r, double scale,
double& maxDelta,
int w, int h)
{
const int size = w*h;
ProjectFunctor_Jacobi functor(p, p2, r, scale, w, h);
Kokkos::parallel_reduce(size, functor, maxDelta);
}

KOKKOS_INLINE_FUNCTION
void init (double& dst) const
{
#ifdef __CUDA_ARCH__
dst = -CUDART_INF;
#else
dst = std::numeric_limits<double>::min();
#endif 
} 

KOKKOS_INLINE_FUNCTION
void operator() (const int& index, double& maxDelta) const
{

int x, y;
index2coord(index,x,y,_w,_h);

double diag = 0.0, offDiag = 0.0;


if (x > 0) {
diag    += _scale;
offDiag -= _scale*_p(x - 1, y    );
}
if (y > 0) {
diag    += _scale;
offDiag -= _scale*_p(x    , y - 1);
}
if (x < _w - 1) {
diag    += _scale;
offDiag -= _scale*_p(x + 1, y    );
}
if (y < _h - 1) {
diag    += _scale;
offDiag -= _scale*_p(x    , y + 1);
}

double newP = ( _r(x,y) - offDiag ) / diag;

maxDelta = fmax(maxDelta, fabs(_p(x,y) - newP));

_p2(x,y) = newP;

} 

KOKKOS_INLINE_FUNCTION
void join (volatile double& dst,
const volatile double& src) const
{
if (dst < src) {
dst = src;
}
} 

Array2d _p, _p2, _r;
double _scale;
int _w,_h;

}; 

#endif 
