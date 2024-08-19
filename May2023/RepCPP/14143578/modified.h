#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/algorithm.h"

#include "magnetic_field.h"



namespace dg
{
namespace geo
{

namespace mod
{


struct Psip: public aCylindricalFunctor<Psip>
{
Psip( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
m_ipoly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
if( m_pred( R,Z))
return m_ipoly( psip);
else
return psip;
}
private:
dg::IPolynomialHeaviside m_ipoly;
std::function<double(double,double)> m_psip;
std::function<bool(double,double)> m_pred;
};
struct PsipR: public aCylindricalFunctor<PsipR>
{
PsipR( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip  = m_psip(R,Z);
double psipR = m_psipR(R,Z);
if( m_pred( R,Z))
return psipR*m_poly( psip);
else
return psipR;
}
private:
dg::PolynomialHeaviside m_poly;
std::function<double(double,double)> m_psip, m_psipR;
std::function<bool(double,double)> m_pred;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
PsipZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
double psipZ = m_psipZ(R,Z);
if( m_pred( R,Z))
return psipZ*m_poly( psip);
else
return psipZ;
}
private:
dg::PolynomialHeaviside m_poly;
std::function<double(double,double)> m_psip, m_psipZ;
std::function<bool(double,double)> m_pred;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
PsipZZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipZZ, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipZ(psipZ), m_psipZZ(psipZZ), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
double psipZ = m_psipZ(R,Z);
double psipZZ = m_psipZZ(R,Z);
if( m_pred( R,Z))
return psipZZ*m_poly( psip) + psipZ*psipZ*m_dpoly(psip);
else
return psipZZ;
}
private:
dg::PolynomialHeaviside m_poly;
dg::DPolynomialHeaviside m_dpoly;
std::function<double(double,double)> m_psip, m_psipZ, m_psipZZ;
std::function<bool(double,double)> m_pred;
};

struct PsipRR: public aCylindricalFunctor<PsipRR>
{
PsipRR( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipRR, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipRR(psipRR), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
double psipR = m_psipR(R,Z);
double psipRR = m_psipRR(R,Z);
if( m_pred( R,Z))
return psipRR*m_poly( psip) + psipR*psipR*m_dpoly(psip);
else
return psipRR;
}
private:
dg::PolynomialHeaviside m_poly;
dg::DPolynomialHeaviside m_dpoly;
std::function<double(double,double)> m_psip, m_psipR, m_psipRR;
std::function<bool(double,double)> m_pred;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
PsipRZ( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, std::function<double(double,double)> psipR, std::function<double(double,double)> psipZ, std::function<double(double,double)> psipRZ, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_dpoly( psi0, alpha, sign), m_psip(psip), m_psipR(psipR), m_psipZ(psipZ), m_psipRZ(psipRZ), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
double psipR = m_psipR(R,Z);
double psipZ = m_psipZ(R,Z);
double psipRZ = m_psipRZ(R,Z);
if( m_pred( R,Z))
return psipRZ*m_poly( psip) + psipR*psipZ*m_dpoly(psip);
else
return psipRZ;
}
private:
dg::PolynomialHeaviside m_poly;
dg::DPolynomialHeaviside m_dpoly;
std::function<double(double,double)> m_psip, m_psipR, m_psipZ, m_psipRZ;
std::function<bool(double,double)> m_pred;
};



static inline dg::geo::CylindricalFunctorsLvl2 createPsip(
const std::function<bool(double,double)> predicate,
const CylindricalFunctorsLvl2& psip,
double psi0, double alpha, double sign = -1)
{
return CylindricalFunctorsLvl2(
mod::Psip(predicate,psip.f(), psi0, alpha, sign),
mod::PsipR(predicate,psip.f(), psip.dfx(), psi0, alpha, sign),
mod::PsipZ(predicate,psip.f(), psip.dfy(), psi0, alpha, sign),
mod::PsipRR(predicate,psip.f(), psip.dfx(), psip.dfxx(), psi0, alpha, sign),
mod::PsipRZ(predicate,psip.f(), psip.dfx(), psip.dfy(), psip.dfxy(), psi0, alpha, sign),
mod::PsipZZ(predicate,psip.f(), psip.dfy(), psip.dfyy(), psi0, alpha, sign));
}

struct DampingRegion : public aCylindricalFunctor<DampingRegion>
{
DampingRegion( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
m_poly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
if( m_pred( R,Z))
return m_poly( m_psip(R,Z));
else
return 0;
}
private:
dg::PolynomialHeaviside m_poly;
std::function<double(double,double)> m_psip;
std::function<bool(double,double)> m_pred;
};
struct MagneticTransition : public aCylindricalFunctor<MagneticTransition>
{
MagneticTransition( std::function<bool(double,double)> predicate, std::function<double(double,double)> psip, double psi0, double alpha, double sign = -1) :
m_dpoly( psi0, alpha, sign), m_psip(psip), m_pred(predicate)
{ }
double do_compute(double R, double Z) const
{
double psip = m_psip(R,Z);
if( m_pred( R,Z))
return m_dpoly( psip);
else
return 0;
}
private:
dg::DPolynomialHeaviside m_dpoly;
std::function<double(double,double)> m_psip;
std::function<bool(double,double)> m_pred;
};
static bool nowhere( double R, double Z){return false;}
static bool everywhere( double R, double Z){return !nowhere(R,Z);}
struct HeavisideZ{
HeavisideZ( double Z_X, int side): m_ZX( Z_X), m_side(side) {}
bool operator()(double R, double Z){
if( Z < m_ZX && m_side <= 0) return true;
if( Z >= m_ZX && m_side > 0) return true;
return false;
}
private:
double m_ZX;
int m_side;
};



struct SetCompose : public aCylindricalFunctor<SetCompose>
{
SetCompose( std::function<double(double,double)> fct_mod,
std::function<double(double,double)> fct1,
std::function<double(double,double)> fct2) :
m_fct1(fct1), m_fct2(fct2), m_fct_mod( fct_mod)
{ }
double do_compute(double R, double Z) const
{
return m_fct_mod( m_fct1(R,Z), m_fct2(R,Z));
}
private:
std::function<double(double,double)> m_fct1, m_fct2, m_fct_mod;
};

struct SetUnion : public aCylindricalFunctor<SetUnion>
{
SetUnion( std::function<double(double,double)> fct1,
std::function<double(double,double)> fct2) :
m_fct1(fct1), m_fct2(fct2)
{ }
double do_compute(double R, double Z) const
{
double f1 = m_fct1(R,Z), f2 = m_fct2( R,Z);
return f1 + f2 - f1*f2;
}
private:
std::function<double(double,double)> m_fct1, m_fct2;
};

struct SetIntersection : public aCylindricalFunctor<SetIntersection>
{
SetIntersection( std::function<double(double,double)> fct1,
std::function<double(double,double)> fct2) :
m_fct1(fct1), m_fct2(fct2)
{ }
double do_compute(double R, double Z) const
{
double f1 = m_fct1(R,Z), f2 = m_fct2( R,Z);
return f1*f2;
}
private:
std::function<double(double,double)> m_fct1, m_fct2;
};

struct SetNot : public aCylindricalFunctor<SetNot>
{
SetNot( std::function<double(double,double)> fct) :
m_fct(fct)
{ }
double do_compute(double R, double Z) const
{
return 1-m_fct(R,Z);
}
private:
std::function<double(double,double)> m_fct;
};


} 

} 
} 
