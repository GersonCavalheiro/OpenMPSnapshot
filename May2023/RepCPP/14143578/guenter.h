#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "magnetic_field.h"


namespace dg
{
namespace geo
{

namespace guenter
{



struct Psip : public aCylindricalFunctor<Psip>
{
Psip(double R_0 ):   R_0(R_0) {}
double do_compute(double R, double Z) const
{
return cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
}
private:
double R_0;
};

struct PsipR : public aCylindricalFunctor<PsipR>
{
PsipR(double R_0 ):   R_0(R_0) {}
double do_compute(double R, double Z) const
{
return -M_PI*0.5*sin(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
}
private:
double R_0;
};

struct PsipRR : public aCylindricalFunctor<PsipRR>
{
PsipRR(double R_0 ):   R_0(R_0) {}
double do_compute(double R, double Z) const
{
return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
}
private:
double R_0;
};

struct PsipZ : public aCylindricalFunctor<PsipZ>

{
PsipZ(double R_0 ):   R_0(R_0) {}
double do_compute(double R, double Z) const
{
return -M_PI*0.5*cos(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
}
private:
double R_0;
};

struct PsipZZ : public aCylindricalFunctor<PsipZZ>
{
PsipZZ(double R_0 ):   R_0(R_0){}
double do_compute(double R, double Z) const
{
return -M_PI*M_PI*0.25*cos(M_PI*0.5*(R-R_0))*cos(M_PI*Z*0.5);
}
private:
double R_0;
};

struct PsipRZ : public aCylindricalFunctor<PsipRZ>
{
PsipRZ(double R_0 ):   R_0(R_0) {}
double do_compute(double R, double Z) const
{
return M_PI*M_PI*0.25*sin(M_PI*0.5*(R-R_0))*sin(M_PI*Z*0.5);
}
private:
double R_0;
};


struct Ipol : public aCylindricalFunctor<Ipol>
{
Ipol( double I_0):   I_0(I_0) {}
double do_compute(double R, double Z) const { return I_0; }
private:
double I_0;
};

struct IpolR : public aCylindricalFunctor<IpolR>
{
IpolR(  ) {}
double do_compute(double R, double Z) const { return 0; }
private:
};

struct IpolZ : public aCylindricalFunctor<IpolZ>
{
IpolZ(  ) {}
double do_compute(double R, double Z) const { return 0; }
private:
};

static inline CylindricalFunctorsLvl2 createPsip( double R_0)
{
return CylindricalFunctorsLvl2( Psip(R_0), PsipR(R_0), PsipZ(R_0),
PsipRR(R_0), PsipRZ(R_0), PsipZZ(R_0));
}
static inline CylindricalFunctorsLvl1 createIpol( double I_0)
{
return CylindricalFunctorsLvl1( Ipol(I_0), IpolR(), IpolZ());
}
} 


static inline dg::geo::TokamakMagneticField createGuenterField( double R_0, double I_0)
{
MagneticFieldParameters params = { 1., 1., 0.,
equilibrium::guenter, modifier::none, description::square};
return TokamakMagneticField( R_0, guenter::createPsip(R_0), guenter::createIpol(I_0), params);
}
} 
}
