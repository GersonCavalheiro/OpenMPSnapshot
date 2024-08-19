#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <boost/math/special_functions.hpp>

#include "dg/algorithm.h"
#include "solovev_parameters.h"
#include "magnetic_field.h"



namespace dg
{
namespace geo
{

namespace taylor
{
typedef dg::geo::solovev::Parameters Parameters; 


struct Psip : public aCylindricalFunctor<Psip>
{ 
Psip( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
}
double do_compute(double R, double Z) const
{
double Rn = R/R0_, Zn = Z/R0_;
double j1_c12 = boost::math::cyl_bessel_j( 1, c_[11]*Rn);
double y1_c12 = boost::math::cyl_neumann(  1, c_[11]*Rn);
double j1_cs = boost::math::cyl_bessel_j( 1, cs_*Rn);
double y1_cs = boost::math::cyl_neumann(  1, cs_*Rn);
return R0_*(
1.0*Rn*j1_c12
+ c_[0]*Rn*y1_c12
+ c_[1]*Rn*j1_cs*cos(c_[10]*Zn)
+ c_[2]*Rn*y1_cs*cos(c_[10]*Zn)
+ c_[3]*cos(c_[11]*sqrt(Rn*Rn+Zn*Zn))
+ c_[4]*cos(c_[11]*Zn)
+ c_[5]*Rn*j1_c12*Zn
+ c_[6]*Rn*y1_c12*Zn
+ c_[7]*Rn*j1_cs*sin(c_[10]*Zn)
+ c_[8]*Rn*y1_cs*sin(c_[10]*Zn)
+ c_[9]*sin(c_[11]*Zn));

}
private:
double R0_, cs_;
std::vector<double> c_;
};


struct PsipR: public aCylindricalFunctor<PsipR>
{
PsipR( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
cs_=sqrt(c_[11]*c_[11]-c_[10]*c_[10]);

}
double do_compute(double R, double Z) const
{
double Rn=R/R0_, Zn=Z/R0_;
double j1_c12R = boost::math::cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(
boost::math::cyl_bessel_j(0, c_[11]*Rn) - boost::math::cyl_bessel_j(2,c_[11]*Rn));
double y1_c12R = boost::math::cyl_neumann(1, c_[11]*Rn) + c_[11]/2.*Rn*(
boost::math::cyl_neumann(0, c_[11]*Rn) - boost::math::cyl_neumann(2,c_[11]*Rn));
double j1_csR = boost::math::cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(
boost::math::cyl_bessel_j(0, cs_*Rn) - boost::math::cyl_bessel_j(2, cs_*Rn));
double y1_csR = boost::math::cyl_neumann(1, cs_*Rn) + cs_/2.*Rn*(
boost::math::cyl_neumann(0, cs_*Rn) - boost::math::cyl_neumann(2, cs_*Rn));
double RZbar = sqrt( Rn*Rn+Zn*Zn);
double cosR = -c_[11]*Rn/RZbar*sin(c_[11]*RZbar);
return  (
1.0*j1_c12R
+ c_[0]*y1_c12R
+ c_[1]*j1_csR*cos(c_[10]*Zn)
+ c_[2]*y1_csR*cos(c_[10]*Zn)
+ c_[3]*cosR
+ c_[5]*j1_c12R*Zn
+ c_[6]*y1_c12R*Zn
+ c_[7]*j1_csR*sin(c_[10]*Zn)
+ c_[8]*y1_csR*sin(c_[10]*Zn) );
}
private:
double R0_, cs_;
std::vector<double> c_;
};

struct PsipRR: public aCylindricalFunctor<PsipRR>
{
PsipRR( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
}
double do_compute(double R, double Z) const
{
double Rn=R/R0_, Zn=Z/R0_;
double j1_c12R = c_[11]*(boost::math::cyl_bessel_j(0, c_[11]*Rn) - Rn*c_[11]*boost::math::cyl_bessel_j(1, c_[11]*Rn));
double y1_c12R = c_[11]*(boost::math::cyl_neumann( 0, c_[11]*Rn) - Rn*c_[11]*boost::math::cyl_neumann(1, c_[11]*Rn));
double j1_csR = cs_*(boost::math::cyl_bessel_j(0, cs_*Rn) - Rn*cs_*boost::math::cyl_bessel_j(1, cs_*Rn));
double y1_csR = cs_*(boost::math::cyl_neumann( 0, cs_*Rn) - Rn*cs_*boost::math::cyl_neumann( 1, cs_*Rn));
double RZbar = sqrt(Rn*Rn+Zn*Zn);
double cosR = -c_[11]/(RZbar*RZbar)*(c_[11]*Rn*Rn*cos(c_[11]*RZbar) +Zn*Zn*sin(c_[11]*RZbar)/RZbar);
return  1./R0_*(
1.0*j1_c12R
+ c_[0]*y1_c12R
+ c_[1]*j1_csR*cos(c_[10]*Zn)
+ c_[2]*y1_csR*cos(c_[10]*Zn)
+ c_[3]*cosR
+ c_[5]*j1_c12R*Zn
+ c_[6]*y1_c12R*Zn
+ c_[7]*j1_csR*sin(c_[10]*Zn)
+ c_[8]*y1_csR*sin(c_[10]*Zn) );
}
private:
double R0_, cs_;
std::vector<double> c_;
};

struct PsipZ: public aCylindricalFunctor<PsipZ>
{
PsipZ( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
}
double do_compute(double R, double Z) const
{
double Rn = R/R0_, Zn = Z/R0_;
double j1_c12 = boost::math::cyl_bessel_j( 1, c_[11]*Rn);
double y1_c12 = boost::math::cyl_neumann(  1, c_[11]*Rn);
double j1_cs = boost::math::cyl_bessel_j( 1, cs_*Rn);
double y1_cs = boost::math::cyl_neumann(  1, cs_*Rn);
return (
- c_[1]*Rn*j1_cs*c_[10]*sin(c_[10]*Zn)
- c_[2]*Rn*y1_cs*c_[10]*sin(c_[10]*Zn)
- c_[3]*c_[11]*Zn/sqrt(Rn*Rn+Zn*Zn)*sin(c_[11]*sqrt(Rn*Rn+Zn*Zn))
- c_[4]*c_[11]*sin(c_[11]*Zn)
+ c_[5]*Rn*j1_c12
+ c_[6]*Rn*y1_c12
+ c_[7]*Rn*j1_cs*c_[10]*cos(c_[10]*Zn)
+ c_[8]*Rn*y1_cs*c_[10]*cos(c_[10]*Zn)
+ c_[9]*c_[11]*cos(c_[11]*Zn));
}
private:
double R0_,cs_;
std::vector<double> c_;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
PsipZZ( solovev::Parameters gp): R0_(gp.R_0), c_(gp.c) {
cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
}
double do_compute(double R, double Z) const
{
double Rn = R/R0_, Zn = Z/R0_;
double j1_cs = boost::math::cyl_bessel_j( 1, cs_*Rn);
double y1_cs = boost::math::cyl_neumann(  1, cs_*Rn);
double RZbar = sqrt(Rn*Rn+Zn*Zn);
double cosZ = -c_[11]/(RZbar*RZbar)*(c_[11]*Zn*Zn*cos(c_[11]*RZbar) +Rn*Rn*sin(c_[11]*RZbar)/RZbar);
return 1./R0_*(
- c_[1]*Rn*j1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)
- c_[2]*Rn*y1_cs*c_[10]*c_[10]*cos(c_[10]*Zn)
+ c_[3]*cosZ
- c_[4]*c_[11]*c_[11]*cos(c_[11]*Zn)
- c_[7]*Rn*j1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
- c_[8]*Rn*y1_cs*c_[10]*c_[10]*sin(c_[10]*Zn)
- c_[9]*c_[11]*c_[11]*sin(c_[11]*Zn));
}
private:
double R0_, cs_;
std::vector<double> c_;
};

struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
PsipRZ( solovev::Parameters gp ): R0_(gp.R_0), c_(gp.c) {
cs_ = sqrt( c_[11]*c_[11]-c_[10]*c_[10]);
}
double do_compute(double R, double Z) const
{
double Rn=R/R0_, Zn=Z/R0_;
double j1_c12R = boost::math::cyl_bessel_j(1, c_[11]*Rn) + c_[11]/2.*Rn*(
boost::math::cyl_bessel_j(0, c_[11]*Rn) - boost::math::cyl_bessel_j(2,c_[11]*Rn));
double y1_c12R = boost::math::cyl_neumann( 1, c_[11]*Rn) + c_[11]/2.*Rn*(
boost::math::cyl_neumann( 0, c_[11]*Rn) - boost::math::cyl_neumann( 2,c_[11]*Rn));
double j1_csR = boost::math::cyl_bessel_j(1, cs_*Rn) + cs_/2.*Rn*(
boost::math::cyl_bessel_j(0, cs_*Rn) - boost::math::cyl_bessel_j(2, cs_*Rn));
double y1_csR = boost::math::cyl_neumann( 1, cs_*Rn) + cs_/2.*Rn*(
boost::math::cyl_neumann( 0, cs_*Rn) - boost::math::cyl_neumann(2, cs_*Rn));
double RZbar = sqrt(Rn*Rn+Zn*Zn);
double cosRZ = -c_[11]*Rn*Zn/(RZbar*RZbar*RZbar)*( c_[11]*RZbar*cos(c_[11]*RZbar) -sin(c_[11]*RZbar) );
return  1./R0_*(
- c_[1]*j1_csR*c_[10]*sin(c_[10]*Zn)
- c_[2]*y1_csR*c_[10]*sin(c_[10]*Zn)
+ c_[3]*cosRZ
+ c_[5]*j1_c12R
+ c_[6]*y1_c12R
+ c_[7]*j1_csR*c_[10]*cos(c_[10]*Zn)
+ c_[8]*y1_csR*c_[10]*cos(c_[10]*Zn) );
}
private:
double R0_, cs_;
std::vector<double> c_;
};


struct Ipol: public aCylindricalFunctor<Ipol>
{
Ipol(  solovev::Parameters gp ): c12_(gp.c[11]), psip_(gp) { }
double do_compute(double R, double Z) const
{
return c12_*psip_(R,Z);

}
private:
double c12_;
Psip psip_;
};

struct IpolR: public aCylindricalFunctor<IpolR>
{
IpolR(  solovev::Parameters gp ): c12_(gp.c[11]), psipR_(gp) { }
double do_compute(double R, double Z) const
{
return c12_*psipR_(R,Z);
}
private:
double c12_;
PsipR psipR_;
};

struct IpolZ: public aCylindricalFunctor<IpolZ>
{
IpolZ(  solovev::Parameters gp ): c12_(gp.c[11]), psipZ_(gp) { }
double do_compute(double R, double Z) const
{
return c12_*psipZ_(R,Z);
}
private:
double c12_;
PsipZ psipZ_;
};

static inline CylindricalFunctorsLvl2 createPsip( solovev::Parameters gp)
{
return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
static inline CylindricalFunctorsLvl1 createIpol( solovev::Parameters gp)
{
return CylindricalFunctorsLvl1( Ipol(gp), IpolR(gp), IpolZ(gp));
}


} 

static inline dg::geo::TokamakMagneticField createTaylorField( dg::geo::solovev::Parameters gp)
{
MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
equilibrium::solovev, modifier::none, str2description.at( gp.description)};
return TokamakMagneticField( gp.R_0, dg::geo::taylor::createPsip(gp), dg::geo::taylor::createIpol(gp), params);
}
} 

}

