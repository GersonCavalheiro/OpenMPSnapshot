#pragma once

#include "magnetic_field.h"

namespace dg{
namespace geo{
namespace toroidal{


static inline CylindricalFunctorsLvl2 createPsip( )
{
CylindricalFunctorsLvl2 psip( Constant(1), Constant(0), Constant(0),Constant(0), Constant(0), Constant(0));
return psip;
}

static inline CylindricalFunctorsLvl1 createIpol( )
{
CylindricalFunctorsLvl1 ipol( Constant(1), Constant(0), Constant(0));
return ipol;
}


}
namespace circular{


struct Psip : public aCylindricalFunctor<Psip>
{ 
Psip( double R0, double a, double b): m_R0(R0), m_a(a), m_b(b) { }
double do_compute(double R, double Z) const
{
return 1. - (R-m_R0)*(R-m_R0)/m_a/m_a - Z*Z/m_b/m_b;
}
private:
double m_R0, m_a, m_b;
};
struct PsipR : public aCylindricalFunctor<PsipR>
{ 
PsipR( double R0, double a): m_R0(R0), m_a(a) { }
double do_compute(double R, double Z) const
{
return -2*(R-m_R0)/m_a/m_a;
}
private:
double m_R0, m_a;
};
struct PsipZ : public aCylindricalFunctor<PsipZ>
{
PsipZ( double b): m_b(b) { }
double do_compute(double R, double Z) const
{
return -2*Z/m_b/m_b;
}
private:
double m_b;
};


static inline CylindricalFunctorsLvl2 createPsip( double R0, double a , double b )
{
return CylindricalFunctorsLvl2( Psip(R0, a, b), PsipR(R0, a), PsipZ(b),
Constant(-2./a/a), Constant(0), Constant(-2./b/b));
}

static inline CylindricalFunctorsLvl1 createIpol( double I0 )
{
CylindricalFunctorsLvl1 ipol( Constant(I0), Constant(0), Constant(0));
return ipol;
}
}


static inline dg::geo::TokamakMagneticField createToroidalField( double R0)
{
MagneticFieldParameters params = { 1., 1., 0.,
equilibrium::circular, modifier::none, description::none};
return TokamakMagneticField( R0, toroidal::createPsip(), toroidal::createIpol(), params);
}

static inline dg::geo::TokamakMagneticField createCircularField( double R0, double I0, double a = 1, double b = 1)
{
MagneticFieldParameters params = { a, 1., 0.,
equilibrium::circular, modifier::none, description::standardO};
return TokamakMagneticField( R0, circular::createPsip(R0, a, b), circular::createIpol(I0), params);
}

}
}
