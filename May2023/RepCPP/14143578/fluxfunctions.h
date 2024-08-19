#pragma once
#include <functional>
#include "dg/algorithm.h"

namespace dg
{
namespace geo
{


template<class real_type>
struct RealCylindricalFunctor
{
RealCylindricalFunctor(){}

template<class BinaryFunctor>
RealCylindricalFunctor( BinaryFunctor f):
m_f(f) {}
real_type operator()( real_type R, real_type Z) const{
return m_f(R,Z);
}
real_type operator()( real_type R, real_type Z, real_type phi) const{
return m_f(R,Z);
}
private:
std::function<real_type(real_type,real_type)> m_f;
};

using CylindricalFunctor = RealCylindricalFunctor<double>;


template<class Derived>
struct aCylindricalFunctor
{

double operator()(double R, double Z) const
{
const Derived& underlying = static_cast<const Derived&>(*this);
return underlying.do_compute(R,Z);
}

double operator()(double R, double Z, double phi)const
{
const Derived& underlying = static_cast<const Derived&>(*this);
return underlying.do_compute(R,Z);
}
#ifndef __CUDACC__ 
private:
friend Derived;
aCylindricalFunctor(){}

aCylindricalFunctor(const aCylindricalFunctor&){}

aCylindricalFunctor& operator=(const aCylindricalFunctor&){return *this;}
#endif 
};


struct Constant: public aCylindricalFunctor<Constant>
{
Constant(double c):c_(c){}
double do_compute(double R,double Z)const{return c_;}
private:
double c_;
};

struct ZCutter : public aCylindricalFunctor<ZCutter>
{
ZCutter(double ZX, int sign = +1): m_heavi( ZX, sign){}
double do_compute(double R, double Z) const {
return m_heavi(Z);
}
private:
dg::Heaviside m_heavi;
};


struct Periodify : public aCylindricalFunctor<Periodify>
{

Periodify( CylindricalFunctor functor, dg::Grid2d g): m_g( g), m_f(functor) {}

Periodify( CylindricalFunctor functor, double R0, double R1, double Z0, double Z1, dg::bc bcx, dg::bc bcy): m_g( R0, R1, Z0, Z1, 3, 10, 10, bcx, bcy), m_f(functor) {}
double do_compute( double R, double Z) const
{
bool negative = false;
m_g.shift( negative, R, Z);
if( negative) return -m_f(R,Z);
return m_f( R, Z);
}
private:
dg::Grid2d m_g;
CylindricalFunctor m_f;
};


struct CylindricalFunctorsLvl1
{
CylindricalFunctorsLvl1(){}

CylindricalFunctorsLvl1(  CylindricalFunctor f,  CylindricalFunctor fx,
CylindricalFunctor fy) : p_{{ f, fx, fy}} {
}
void reset( CylindricalFunctor f, CylindricalFunctor fx, CylindricalFunctor fy)
{
p_[0] = f;
p_[1] = fx;
p_[2] = fy;
}
const CylindricalFunctor& f()const{return p_[0];}
const CylindricalFunctor& dfx()const{return p_[1];}
const CylindricalFunctor& dfy()const{return p_[2];}
private:
std::array<CylindricalFunctor,3> p_;
};



struct CylindricalFunctorsLvl2
{
CylindricalFunctorsLvl2(){}

CylindricalFunctorsLvl2(  CylindricalFunctor f,  CylindricalFunctor fx,
CylindricalFunctor fy,   CylindricalFunctor fxx,
CylindricalFunctor fxy,  CylindricalFunctor fyy):
f0(f,fx,fy), f1(fxx,fxy,fyy)
{ }
void reset( CylindricalFunctor f, CylindricalFunctor fx,
CylindricalFunctor fy, CylindricalFunctor fxx,
CylindricalFunctor fxy, CylindricalFunctor fyy)
{
f0.reset( f,fx,fy), f1.reset(fxx,fxy,fyy);
}
operator CylindricalFunctorsLvl1 ()const {return f0;}
const CylindricalFunctor& f()const{return f0.f();}
const CylindricalFunctor& dfx()const{return f0.dfx();}
const CylindricalFunctor& dfy()const{return f0.dfy();}
const CylindricalFunctor& dfxx()const{return f1.f();}
const CylindricalFunctor& dfxy()const{return f1.dfx();}
const CylindricalFunctor& dfyy()const{return f1.dfy();}
private:
CylindricalFunctorsLvl1 f0,f1;
};



static inline int findCriticalPoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
std::array<double, 2> X{ {0,0} }, XN(X), X_OLD(X);
X[0] = RC, X[1] = ZC;
double eps = 1e10, eps_old= 2e10;
unsigned counter = 0; 
double psipRZ = psi.dfxy()(X[0], X[1]);
double psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
double psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
double D0 =  (psipZZ*psipRR - psipRZ*psipRZ);
if(D0 == 0) 
{
X[0] *= 1.0001, X[1]*=1.0001;
psipRZ = psi.dfxy()(X[0], X[1]);
psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
D0 =  (psipZZ*psipRR - psipRZ*psipRZ);
}
double Dinv = 1./D0;
while( (eps < eps_old || eps > 1e-7) && eps > 1e-10 && counter < 100)
{
XN[0] = X[0] - Dinv*(psipZZ*psipR - psipRZ*psipZ);
XN[1] = X[1] - Dinv*(-psipRZ*psipR + psipRR*psipZ);
XN.swap(X);
eps = sqrt( (X[0]-X_OLD[0])*(X[0]-X_OLD[0]) + (X[1]-X_OLD[1])*(X[1]-X_OLD[1]));
X_OLD = X; eps_old= eps;
psipRZ = psi.dfxy()(X[0], X[1]);
psipRR = psi.dfxx()(X[0], X[1]), psipZZ = psi.dfyy()(X[0],X[1]);
psipR  = psi.dfx()(X[0], X[1]), psipZ = psi.dfy()(X[0], X[1]);
D0 = (psipZZ*psipRR - psipRZ*psipRZ);
Dinv = 1./D0;
if( D0 == 0) break;
counter++;
}
if ( counter >= 100 || D0 == 0|| std::isnan( Dinv) )
return 0;
RC = X[0], ZC = X[1];
if( Dinv > 0 &&  psipRR > 0)
return 1; 
if( Dinv > 0 &&  psipRR < 0)
return 2; 
return 3; 
}


static inline int findOpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
int point = findCriticalPoint( psi, RC, ZC);
if( point == 3 || point == 0 )
throw dg::Error(dg::Message(_ping_)<<"There is no O-point near "<<RC<<" "<<ZC);
return point;
}


static inline void findXpoint( const CylindricalFunctorsLvl2& psi, double& RC, double& ZC)
{
int point = findCriticalPoint( psi, RC, ZC);
if( point != 3)
throw dg::Error(dg::Message(_ping_)<<"There is no X-point near "<<RC<<" "<<ZC);
}


struct CylindricalSymmTensorLvl1
{

CylindricalSymmTensorLvl1( ){
reset( Constant(1), Constant(0), Constant(1), Constant(0), Constant(0));
}

CylindricalSymmTensorLvl1(  CylindricalFunctor chi_xx,
CylindricalFunctor chi_xy,   CylindricalFunctor chi_yy,
CylindricalFunctor divChiX,  CylindricalFunctor divChiY) :
p_{{ chi_xx,chi_xy,chi_yy,divChiX,divChiY}}
{
}
void reset( CylindricalFunctor chi_xx, CylindricalFunctor chi_xy,
CylindricalFunctor chi_yy, CylindricalFunctor divChiX,
CylindricalFunctor divChiY)
{
p_[0] = chi_xx;
p_[1] = chi_xy;
p_[2] = chi_yy;
p_[3] = divChiX;
p_[4] = divChiY;
}
const CylindricalFunctor& xx()const{return p_[0];}
const CylindricalFunctor& xy()const{return p_[1];}
const CylindricalFunctor& yy()const{return p_[2];}
const CylindricalFunctor& divX()const{return p_[3];}
const CylindricalFunctor& divY()const{return p_[4];}
private:
std::array<CylindricalFunctor,5> p_;
};

struct CylindricalVectorLvl0
{
CylindricalVectorLvl0(){}
CylindricalVectorLvl0(  CylindricalFunctor v_x,
CylindricalFunctor v_y,
CylindricalFunctor v_z): p_{{v_x, v_y, v_z}}{}
void reset(  CylindricalFunctor v_x,  CylindricalFunctor v_y,
CylindricalFunctor v_z)
{
p_[0] = v_x;
p_[1] = v_y;
p_[2] = v_z;
}
const CylindricalFunctor& x()const{return p_[0];}
const CylindricalFunctor& y()const{return p_[1];}
const CylindricalFunctor& z()const{return p_[2];}
private:
std::array<CylindricalFunctor,3> p_;
};


struct CylindricalVectorLvl1
{
CylindricalVectorLvl1(){}
CylindricalVectorLvl1(  CylindricalFunctor v_x,
CylindricalFunctor v_y,
CylindricalFunctor v_z,
CylindricalFunctor div,
CylindricalFunctor divvvz
): f0{v_x, v_y, v_z},
m_div(div), m_divvvz(divvvz) {}
void reset(  CylindricalFunctor v_x,
CylindricalFunctor v_y,
CylindricalFunctor v_z,
CylindricalFunctor div,
CylindricalFunctor divvvz
)
{
f0.reset( v_x,v_y,v_z);
m_div = div;
m_divvvz = divvvz;
}
operator CylindricalVectorLvl0 ()const {return f0;}
const CylindricalFunctor& x()const{return f0.x();}
const CylindricalFunctor& y()const{return f0.y();}
const CylindricalFunctor& z()const{return f0.z();}
const CylindricalFunctor& div()const{return m_div;}
const CylindricalFunctor& divvvz()const{return m_divvvz;}
private:
CylindricalVectorLvl0 f0;
CylindricalFunctor m_div, m_divvvz;
};


struct ScalarProduct : public aCylindricalFunctor<ScalarProduct>
{
ScalarProduct( CylindricalVectorLvl0 v, CylindricalVectorLvl0 w) : m_v(v), m_w(w){}
double do_compute( double R, double Z) const
{
return m_v.x()(R,Z)*m_w.x()(R,Z)
+ m_v.y()(R,Z)*m_w.y()(R,Z)
+ m_v.z()(R,Z)*m_w.z()(R,Z);
}
private:
CylindricalVectorLvl0 m_v, m_w;
};


struct SquareNorm : public aCylindricalFunctor<SquareNorm>
{
SquareNorm( CylindricalVectorLvl0 v, CylindricalVectorLvl0 w) : m_s(v, w){}
double do_compute( double R, double Z) const
{
return sqrt(m_s(R,Z));
}
private:
ScalarProduct m_s;
};



template<class Geometry3d>
dg::SparseTensor<dg::get_host_vector<Geometry3d>> createAlignmentTensor(
const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
using host_vector = dg::get_host_vector<Geometry3d>;
SparseTensor<host_vector> t;
std::array<host_vector,3> bt;
dg::pushForward( bhat.x(), bhat.y(), bhat.z(), bt[0], bt[1], bt[2], g);
std::vector<host_vector> chi(6, dg::evaluate( dg::zero,g));
dg::blas1::pointwiseDot( bt[0], bt[0], chi[0]);
dg::blas1::pointwiseDot( bt[0], bt[1], chi[1]);
dg::blas1::pointwiseDot( bt[0], bt[2], chi[2]);
dg::blas1::pointwiseDot( bt[1], bt[1], chi[3]);
dg::blas1::pointwiseDot( bt[1], bt[2], chi[4]);
dg::blas1::pointwiseDot( bt[2], bt[2], chi[5]);
t.idx(0,0) = 0, t.idx(0,1) = t.idx(1,0) = 1,
t.idx(0,2) = t.idx(2,0) = 2;
t.idx(1,1) = 3, t.idx(1,2) = t.idx(2,1) = 4;
t.idx(2,2) = 5;
t.values() = chi;
return t;
}

template<class Geometry3d>
dg::SparseTensor<dg::get_host_vector<Geometry3d>> createProjectionTensor(
const dg::geo::CylindricalVectorLvl0& bhat, const Geometry3d& g)
{
using host_vector = dg::get_host_vector<Geometry3d>;
dg::SparseTensor<host_vector> t = dg::geo::createAlignmentTensor( bhat, g);
dg::SparseTensor<host_vector> m = g.metric();
dg::blas1::axpby( 1., m.value(0,0), -1., t.values()[0]);
dg::blas1::axpby( 1., m.value(0,1), -1., t.values()[1]);
dg::blas1::axpby( 1., m.value(0,2), -1., t.values()[2]);
dg::blas1::axpby( 1., m.value(1,1), -1., t.values()[3]);
dg::blas1::axpby( 1., m.value(1,2), -1., t.values()[4]);
dg::blas1::axpby( 1., m.value(2,2), -1., t.values()[5]);
return t;
}

}
}
