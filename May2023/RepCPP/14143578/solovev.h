#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/algorithm.h"
#include "solovev_parameters.h"
#include "magnetic_field.h"
#include "modified.h"



namespace dg
{
namespace geo
{

namespace solovev
{


struct Psip: public aCylindricalFunctor<Psip>
{

Psip( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
Zn = Z/m_R0; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
lgRn= log(Rn);
return   m_R0*m_pp*( Rn4/8.+ m_A * ( 1./2.* Rn2* lgRn-(Rn4)/8.)
+ m_c[0]  
+ m_c[1]  *Rn2
+ m_c[2]  *(Zn2 - Rn2 * lgRn )
+ m_c[3]  *(Rn4 - 4.* Rn2*Zn2 )
+ m_c[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
+ m_c[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 )
+ m_c[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn
-140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
+ m_c[7]  *Zn
+ m_c[8]  *Rn2*Zn
+ m_c[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
+ m_c[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
+ m_c[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)
);
}
private:
double m_R0, m_A, m_pp;
std::vector<double> m_c;
};


struct PsipR: public aCylindricalFunctor<PsipR>
{
PsipR( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2;
Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
lgRn= log(Rn);
return   m_pp*(Rn3/2. + (Rn/2. - Rn3/2. + Rn*lgRn)* m_A +
2.* Rn* m_c[1] + (-Rn - 2.* Rn*lgRn)* m_c[2] + (4.*Rn3 - 8.* Rn *Zn2)* m_c[3] +
(3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* m_c[4]
+ (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*m_c[5]
+ (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn +
720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* m_c[6] +
2.* Rn *Zn *m_c[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* m_c[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* m_c[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn -
160.* Rn *Zn3*lgRn) *m_c[11]
);
}
private:
double m_R0, m_A, m_pp;
std::vector<double> m_c;
};

struct PsipRR: public aCylindricalFunctor<PsipRR>
{
PsipRR( Parameters gp ): m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_c(gp.c) {}
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
lgRn= log(Rn);
return   m_pp/m_R0*( (3.* Rn2)/2. + (3./2. - (3. *Rn2)/2. +lgRn) *m_A +  2.* m_c[1] + (-3. - 2.*lgRn)* m_c[2] + (12. *Rn2 - 8. *Zn2) *m_c[3] +
(21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* m_c[4]
+ (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*m_c[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn +
2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* m_c[6] +
2.* Zn* m_c[8] + (-9. *Zn - 6.* Zn*lgRn) *m_c[9]
+   (36. *Rn2* Zn - 8. *Zn3) *m_c[10]
+   (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* m_c[11]);
}
private:
double m_R0, m_A, m_pp;
std::vector<double> m_c;
};

struct PsipZ: public aCylindricalFunctor<PsipZ>
{
PsipZ( Parameters gp ): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
Zn = Z/m_R0; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2;
lgRn= log(Rn);

return   m_pp*(2.* Zn* m_c[2]
-  8. *Rn2* Zn* m_c[3] +
((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *m_c[4]
+ ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* m_c[5]
+ (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* m_c[6]
+ m_c[7]
+ Rn2 * m_c[8]
+ (3. *Zn2 - 3. *Rn2*lgRn)* m_c[9]
+ (3. *Rn4 - 12. *Rn2 *Zn2) *m_c[10]
+ ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* m_c[11]);

}
private:
double m_R0, m_pp;
std::vector<double> m_c;
};

struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
PsipZZ( Parameters gp): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
lgRn= log(Rn);
return   m_pp/m_R0*( 2.* m_c[2] - 8. *Rn2* m_c[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *m_c[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *m_c[5]
+ (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* m_c[6] + 6.* Zn* m_c[9] -  24. *Rn2 *Zn *m_c[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *m_c[11]);
}
private:
double m_R0, m_pp;
std::vector<double> m_c;
};

struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
PsipRZ( Parameters gp ): m_R0(gp.R_0), m_pp(gp.pp), m_c(gp.c) { }
double do_compute(double R, double Z) const
{
double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
Rn = R/m_R0; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;
Zn = Z/m_R0; Zn2 =Zn*Zn; Zn3 = Zn2*Zn;
lgRn= log(Rn);
return   m_pp/m_R0*(
-16.* Rn* Zn* m_c[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* m_c[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* m_c[5]
+ (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *m_c[6] +  2.* Rn* m_c[8] + (-3.* Rn - 6.* Rn*lgRn)* m_c[9]
+ (12. *Rn3 - 24.* Rn *Zn2) *m_c[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* m_c[11]
);
}
private:
double m_R0, m_pp;
std::vector<double> m_c;
};


struct Ipol: public aCylindricalFunctor<Ipol>
{

Ipol( Parameters gp, std::function<double(double,double)> psip ):  m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip) {
if( gp.pp == 0.)
m_pp = 1.; 
}
double do_compute(double R, double Z) const
{
return m_pi*sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.);
}
private:
double m_R0, m_A, m_pp, m_pi;
std::function<double(double,double)> m_psip;
};

struct IpolR: public aCylindricalFunctor<IpolR>
{

IpolR(  Parameters gp, std::function<double(double,double)> psip, std::function<double(double,double)> psipR ):
m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip), m_psipR(psipR) {
if( gp.pp == 0.)
m_pp = 1.; 
}
double do_compute(double R, double Z) const
{
return -m_pi/sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.)*(m_A*m_psipR(R,Z)/m_R0/m_pp);
}
private:
double m_R0, m_A, m_pp, m_pi;
std::function<double(double,double)> m_psip, m_psipR;
};

struct IpolZ: public aCylindricalFunctor<IpolZ>
{

IpolZ(  Parameters gp, std::function<double(double,double)> psip, std::function<double(double,double)> psipZ ):
m_R0(gp.R_0), m_A(gp.A), m_pp(gp.pp), m_pi(gp.pi), m_psip(psip), m_psipZ(psipZ) {
if( gp.pp == 0.)
m_pp = 1.; 
}
double do_compute(double R, double Z) const
{
return -m_pi/sqrt(-2.*m_A* m_psip(R,Z) /m_R0/m_pp + 1.)*(m_A*m_psipZ(R,Z)/m_R0/m_pp);
}
private:
double m_R0, m_A, m_pp, m_pi;
std::function<double(double,double)> m_psip, m_psipZ;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( const Parameters& gp)
{
return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( const Parameters& gp, const CylindricalFunctorsLvl1& psip)
{
return CylindricalFunctorsLvl1(
solovev::Ipol(gp, psip.f()),
solovev::IpolR(gp,psip.f(), psip.dfx()),
solovev::IpolZ(gp,psip.f(), psip.dfy()));
}


} 


static inline dg::geo::TokamakMagneticField createSolovevField(
dg::geo::solovev::Parameters gp)
{
MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
equilibrium::solovev, modifier::none, str2description.at( gp.description)};
return TokamakMagneticField( gp.R_0, solovev::createPsip(gp),
solovev::createIpol(gp, solovev::createPsip(gp)), params);
}

static inline dg::geo::TokamakMagneticField createModifiedSolovevField(
dg::geo::solovev::Parameters gp, double psi0, double alpha, double sign = -1)
{
MagneticFieldParameters params = { gp.a, gp.elongation, gp.triangularity,
equilibrium::solovev, modifier::heaviside, str2description.at( gp.description)};
return TokamakMagneticField( gp.R_0,
mod::createPsip( mod::everywhere, solovev::createPsip(gp), psi0, alpha, sign),
solovev::createIpol( gp, mod::createPsip( mod::everywhere, solovev::createPsip(gp), psi0, alpha, sign)),
params);
}

} 
} 

