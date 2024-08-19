#pragma once

#include <map>
#include "fluxfunctions.h"


namespace dg
{
namespace geo
{



enum class equilibrium
{
solovev, 
taylor, 
polynomial, 
guenter, 
toroidal, 
circular 
};
enum class modifier
{
none, 
heaviside, 
sol_pfr 
};

enum class description
{
standardO, 
standardX, 
doubleX, 
none, 
square, 
centeredX 
};
static const std::map<std::string, equilibrium> str2equilibrium{
{"solovev", equilibrium::solovev},
{"taylor", equilibrium::taylor},
{"polynomial", equilibrium::polynomial},
{"guenter", equilibrium::guenter},
{"toroidal", equilibrium::toroidal},
{"circular", equilibrium::circular}
};
static const std::map<std::string, modifier> str2modifier{
{"none", modifier::none},
{"heaviside", modifier::heaviside},
{"sol_pfr", modifier::sol_pfr}
};
static const std::map<std::string, description> str2description{
{"standardO", description::standardO},
{"standardX", description::standardX},
{"doubleX", description::doubleX},
{"square", description::square},
{"none", description::none},
{"centeredX", description::centeredX}
};


struct MagneticFieldParameters
{

MagneticFieldParameters( ){
m_a = 1, m_elongation = 1, m_triangularity = 0;
m_equilibrium = equilibrium::toroidal;
m_modifier = modifier::none;
m_description = description::none;
}

MagneticFieldParameters( double a, double elongation, double triangularity,
equilibrium equ, modifier mod, description des): m_a(a),
m_elongation(elongation),
m_triangularity( triangularity),
m_equilibrium( equ),
m_modifier(mod), m_description( des){}

double a() const{return m_a;}

double elongation() const{return m_elongation;}

double triangularity() const{return m_triangularity;}
equilibrium getEquilibrium() const{return m_equilibrium;}
modifier getModifier() const{return m_modifier;}
description getDescription() const{return m_description;}
private:
double m_a,
m_elongation,
m_triangularity;
equilibrium m_equilibrium;
modifier m_modifier;
description m_description;
};


struct TokamakMagneticField
{
TokamakMagneticField(){}
TokamakMagneticField( double R0, const CylindricalFunctorsLvl2& psip, const
CylindricalFunctorsLvl1& ipol , MagneticFieldParameters gp
): m_R0(R0), m_psip(psip), m_ipol(ipol), m_params(gp){}
void set( double R0, const CylindricalFunctorsLvl2& psip, const
CylindricalFunctorsLvl1& ipol , MagneticFieldParameters gp)
{
m_R0=R0;
m_psip=psip;
m_ipol=ipol;
m_params = gp;
}
double R0()const {return m_R0;}
const CylindricalFunctor& psip()const{return m_psip.f();}
const CylindricalFunctor& psipR()const{return m_psip.dfx();}
const CylindricalFunctor& psipZ()const{return m_psip.dfy();}
const CylindricalFunctor& psipRR()const{return m_psip.dfxx();}
const CylindricalFunctor& psipRZ()const{return m_psip.dfxy();}
const CylindricalFunctor& psipZZ()const{return m_psip.dfyy();}
const CylindricalFunctor& ipol()const{return m_ipol.f();}
const CylindricalFunctor& ipolR()const{return m_ipol.dfx();}
const CylindricalFunctor& ipolZ()const{return m_ipol.dfy();}

const CylindricalFunctorsLvl2& get_psip() const{return m_psip;}
const CylindricalFunctorsLvl1& get_ipol() const{return m_ipol;}

const MagneticFieldParameters& params() const{return m_params;}

private:
double m_R0;
CylindricalFunctorsLvl2 m_psip;
CylindricalFunctorsLvl1 m_ipol;
MagneticFieldParameters m_params;
};

static inline CylindricalFunctorsLvl1 periodify( const CylindricalFunctorsLvl1& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
{
return CylindricalFunctorsLvl1(
Periodify( in.f(),   R0, R1, Z0, Z1, bcx, bcy),
Periodify( in.dfx(), R0, R1, Z0, Z1, inverse(bcx), bcy),
Periodify( in.dfy(), R0, R1, Z0, Z1, bcx, inverse(bcy)));
}
static inline CylindricalFunctorsLvl2 periodify( const CylindricalFunctorsLvl2& in, double R0, double R1, double Z0, double Z1, bc bcx, bc bcy)
{
return CylindricalFunctorsLvl2(
Periodify( in.f(),   R0, R1, Z0, Z1, bcx, bcy),
Periodify( in.dfx(), R0, R1, Z0, Z1, inverse(bcx), bcy),
Periodify( in.dfy(), R0, R1, Z0, Z1, bcx, inverse(bcy)),
Periodify( in.dfxx(), R0, R1, Z0, Z1, bcx, bcy),
Periodify( in.dfxy(), R0, R1, Z0, Z1, inverse(bcx), inverse(bcy)),
Periodify( in.dfyy(), R0, R1, Z0, Z1, bcx, bcy));
}

static inline TokamakMagneticField periodify( const TokamakMagneticField& mag, double R0, double R1, double Z0, double Z1, dg::bc bcx, dg::bc bcy)
{
return TokamakMagneticField( mag.R0(),
periodify( mag.get_psip(), R0, R1, Z0, Z1, bcx, bcy),
periodify( mag.get_ipol(), R0, R1, Z0, Z1, bcx, bcy), mag.params());
}

struct LaplacePsip : public aCylindricalFunctor<LaplacePsip>
{
LaplacePsip( const TokamakMagneticField& mag): m_mag(mag)  { }
double do_compute(double R, double Z) const
{
return m_mag.psipR()(R,Z)/R+ m_mag.psipRR()(R,Z) + m_mag.psipZZ()(R,Z);
}
private:
TokamakMagneticField m_mag;
};

struct Bmodule : public aCylindricalFunctor<Bmodule>
{
Bmodule( const TokamakMagneticField& mag): m_mag(mag)  { }
double do_compute(double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
return m_mag.R0()/R*sqrt(ipol*ipol+psipR*psipR +psipZ*psipZ);
}
private:
TokamakMagneticField m_mag;
};


struct InvB : public aCylindricalFunctor<InvB>
{
InvB(  const TokamakMagneticField& mag): m_mag(mag){ }
double do_compute(double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
return R/(m_mag.R0()*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
}
private:
TokamakMagneticField m_mag;
};


struct LnB : public aCylindricalFunctor<LnB>
{
LnB(const TokamakMagneticField& mag): m_mag(mag) { }
double do_compute(double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
return log(m_mag.R0()/R*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
}
private:
TokamakMagneticField m_mag;
};


struct BR: public aCylindricalFunctor<BR>
{
BR(const TokamakMagneticField& mag): m_invB(mag), m_mag(mag) { }
double do_compute(double R, double Z) const
{
double Rn = R/m_mag.R0();
double invB = m_invB(R,Z);
return -1./R/invB + invB/Rn/Rn*(m_mag.ipol()(R,Z)*m_mag.ipolR()(R,Z) + m_mag.psipR()(R,Z)*m_mag.psipRR()(R,Z) + m_mag.psipZ()(R,Z)*m_mag.psipRZ()(R,Z));
}
private:
InvB m_invB;
TokamakMagneticField m_mag;
};


struct BZ: public aCylindricalFunctor<BZ>
{
BZ(const TokamakMagneticField& mag ): m_mag(mag), m_invB(mag) { }
double do_compute(double R, double Z) const
{
double Rn = R/m_mag.R0();
return (m_invB(R,Z)/Rn/Rn)*(m_mag.ipol()(R,Z)*m_mag.ipolZ()(R,Z) + m_mag.psipR()(R,Z)*m_mag.psipRZ()(R,Z) + m_mag.psipZ()(R,Z)*m_mag.psipZZ()(R,Z));
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
};

struct CurvatureNablaBR: public aCylindricalFunctor<CurvatureNablaBR>
{
CurvatureNablaBR(const TokamakMagneticField& mag, int sign): m_invB(mag), m_bZ(mag) {
if( sign >0)
m_sign = +1.;
else
m_sign = -1;
}
double do_compute( double R, double Z) const
{
return -m_sign*m_invB(R,Z)*m_invB(R,Z)*m_bZ(R,Z);
}
private:
double m_sign;
InvB m_invB;
BZ m_bZ;
};

struct CurvatureNablaBZ: public aCylindricalFunctor<CurvatureNablaBZ>
{
CurvatureNablaBZ( const TokamakMagneticField& mag, int sign): m_invB(mag), m_bR(mag) {
if( sign >0)
m_sign = +1.;
else
m_sign = -1;
}
double do_compute( double R, double Z) const
{
return m_sign*m_invB(R,Z)*m_invB(R,Z)*m_bR(R,Z);
}
private:
double m_sign;
InvB m_invB;
BR m_bR;
};

struct CurvatureKappaR: public aCylindricalFunctor<CurvatureKappaR>
{
CurvatureKappaR( ){ }
CurvatureKappaR( const TokamakMagneticField& mag, int sign = +1){ }
double do_compute( double R, double Z) const
{
return  0.;
}
private:
};

struct CurvatureKappaZ: public aCylindricalFunctor<CurvatureKappaZ>
{
CurvatureKappaZ( const TokamakMagneticField& mag, int sign): m_invB(mag) {
if( sign >0)
m_sign = +1.;
else
m_sign = -1;
}
double do_compute( double R, double Z) const
{
return -m_sign*m_invB(R,Z)/R;
}
private:
double m_sign;
InvB m_invB;
};

struct DivCurvatureKappa: public aCylindricalFunctor<DivCurvatureKappa>
{
DivCurvatureKappa( const TokamakMagneticField& mag, int sign): m_invB(mag), m_bZ(mag){
if( sign >0)
m_sign = +1.;
else
m_sign = -1;
}
double do_compute( double R, double Z) const
{
return m_sign*m_bZ(R,Z)*m_invB(R,Z)*m_invB(R,Z)/R;
}
private:
double m_sign;
InvB m_invB;
BZ m_bZ;
};

struct DivCurvatureNablaB: public aCylindricalFunctor<DivCurvatureNablaB>
{
DivCurvatureNablaB( const TokamakMagneticField& mag, int sign): m_div(mag, sign){ }
double do_compute( double R, double Z) const
{
return -m_div(R,Z);
}
private:
DivCurvatureKappa m_div;
};
struct TrueCurvatureNablaBR: public aCylindricalFunctor<TrueCurvatureNablaBR>
{
TrueCurvatureNablaBR(const TokamakMagneticField& mag): m_R0(mag.R0()), m_mag(mag), m_invB(mag), m_bZ(mag) { }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z), ipol = m_mag.ipol()(R,Z);
return -invB*invB*invB*ipol*m_R0/R*m_bZ(R,Z);
}
private:
double m_R0;
TokamakMagneticField m_mag;
InvB m_invB;
BZ m_bZ;
};

struct TrueCurvatureNablaBZ: public aCylindricalFunctor<TrueCurvatureNablaBZ>
{
TrueCurvatureNablaBZ(const TokamakMagneticField& mag): m_R0(mag.R0()), m_mag(mag), m_invB(mag), m_bR(mag) { }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z), ipol = m_mag.ipol()(R,Z);
return invB*invB*invB*ipol*m_R0/R*m_bR(R,Z);
}
private:
double m_R0;
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
};

struct TrueCurvatureNablaBP: public aCylindricalFunctor<TrueCurvatureNablaBP>
{
TrueCurvatureNablaBP(const TokamakMagneticField& mag): m_mag(mag), m_invB(mag),m_bR(mag), m_bZ(mag) { }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB*invB/R/R*(m_mag.psipZ()(R,Z)*m_bZ(R,Z) + m_mag.psipR()(R,Z)*m_bR(R,Z));
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
BZ m_bZ;
};

struct TrueCurvatureKappaR: public aCylindricalFunctor<TrueCurvatureKappaR>
{
TrueCurvatureKappaR( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bZ(mag){ }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB/R*(m_mag.ipolZ()(R,Z) - m_mag.ipol()(R,Z)*invB*m_bZ(R,Z));
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BZ m_bZ;
};

struct TrueCurvatureKappaZ: public aCylindricalFunctor<TrueCurvatureKappaZ>
{
TrueCurvatureKappaZ( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bR(mag){ }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB/R*( - m_mag.ipolR()(R,Z) + m_mag.ipol()(R,Z)*invB*m_bR(R,Z));
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
};
struct TrueCurvatureKappaP: public aCylindricalFunctor<TrueCurvatureKappaP>
{
TrueCurvatureKappaP( const TokamakMagneticField& mag):m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag){ }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB/R/R*(
+ invB*m_mag.psipZ()(R,Z)*m_bZ(R,Z) + invB *m_mag.psipR()(R,Z)*m_bR(R,Z)
+ m_mag.psipR()(R,Z)/R - m_mag.psipRR()(R,Z) - m_mag.psipZZ()(R,Z));
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
BZ m_bZ;
};

struct TrueDivCurvatureKappa: public aCylindricalFunctor<TrueDivCurvatureKappa>
{
TrueDivCurvatureKappa( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag){}
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB*invB/R*( m_mag.ipolR()(R,Z)*m_bZ(R,Z) - m_mag.ipolZ()(R,Z)*m_bR(R,Z) );
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
BZ m_bZ;
};

struct TrueDivCurvatureNablaB: public aCylindricalFunctor<TrueDivCurvatureNablaB>
{
TrueDivCurvatureNablaB( const TokamakMagneticField& mag): m_div(mag){}
double do_compute( double R, double Z) const {
return - m_div(R,Z);
}
private:
TrueDivCurvatureKappa m_div;
};


struct GradLnB: public aCylindricalFunctor<GradLnB>
{
GradLnB( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bR(mag), m_bZ(mag) { }
double do_compute( double R, double Z) const
{
double invB = m_invB(R,Z);
return m_mag.R0()*invB*invB*(m_bR(R,Z)*m_mag.psipZ()(R,Z)-m_bZ(R,Z)*m_mag.psipR()(R,Z))/R ;
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_bR;
BZ m_bZ;
};

struct Divb: public aCylindricalFunctor<Divb>
{
Divb( const TokamakMagneticField& mag): m_gradLnB(mag) { }
double do_compute( double R, double Z) const
{
return -m_gradLnB(R,Z);
}
private:
GradLnB m_gradLnB;
};

struct BFieldP: public aCylindricalFunctor<BFieldP>
{
BFieldP( const TokamakMagneticField& mag): m_mag(mag){}
double do_compute( double R, double Z) const
{
return m_mag.R0()*m_mag.ipol()(R,Z)/R/R;
}
private:

TokamakMagneticField m_mag;
};

struct BFieldR: public aCylindricalFunctor<BFieldR>
{
BFieldR( const TokamakMagneticField& mag): m_mag(mag){}
double do_compute( double R, double Z) const
{
return  m_mag.R0()/R*m_mag.psipZ()(R,Z);
}
private:
TokamakMagneticField m_mag;

};

struct BFieldZ: public aCylindricalFunctor<BFieldZ>
{
BFieldZ( const TokamakMagneticField& mag): m_mag(mag){}
double do_compute( double R, double Z) const
{
return -m_mag.R0()/R*m_mag.psipR()(R,Z);
}
private:
TokamakMagneticField m_mag;
};

struct BFieldT: public aCylindricalFunctor<BFieldT>
{
BFieldT( const TokamakMagneticField& mag):  m_R0(mag.R0()), m_fieldR(mag), m_fieldZ(mag){}
double do_compute(double R, double Z) const
{
double r2 = (R-m_R0)*(R-m_R0) + Z*Z;
return m_fieldR(R,Z)*(-Z/r2) + m_fieldZ(R,Z)*(R-m_R0)/r2;
}
private:
double m_R0;
BFieldR m_fieldR;
BFieldZ m_fieldZ;
};

struct BHatR: public aCylindricalFunctor<BHatR>
{
BHatR( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
double do_compute( double R, double Z) const
{
return  m_invB(R,Z)*m_mag.R0()/R*m_mag.psipZ()(R,Z);
}
private:
TokamakMagneticField m_mag;
InvB m_invB;

};

struct BHatZ: public aCylindricalFunctor<BHatZ>
{
BHatZ( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
double do_compute( double R, double Z) const
{
return  -m_invB(R,Z)*m_mag.R0()/R*m_mag.psipR()(R,Z);
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
};

struct BHatP: public aCylindricalFunctor<BHatP>
{
BHatP( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag){ }
double do_compute( double R, double Z) const
{
return m_invB(R,Z)*m_mag.R0()*m_mag.ipol()(R,Z)/R/R;
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
};



inline CylindricalVectorLvl1 createEPhi( int sign ){
if( sign > 0)
return CylindricalVectorLvl1( Constant(0), Constant(0), [](double x, double y){ return 1./x;}, Constant(0), Constant(0));
return CylindricalVectorLvl1( Constant(0), Constant(0), [](double x, double y){ return -1./x;}, Constant(0), Constant(0));
}

inline CylindricalVectorLvl0 createCurvatureNablaB( const TokamakMagneticField& mag, int sign){
return CylindricalVectorLvl0( CurvatureNablaBR(mag, sign), CurvatureNablaBZ(mag, sign), Constant(0));
}

inline CylindricalVectorLvl0 createCurvatureKappa( const TokamakMagneticField& mag, int sign){
return CylindricalVectorLvl0( CurvatureKappaR(mag, sign), CurvatureKappaZ(mag, sign), Constant(0));
}

inline CylindricalVectorLvl0 createTrueCurvatureKappa( const TokamakMagneticField& mag){
return CylindricalVectorLvl0( TrueCurvatureKappaR(mag), TrueCurvatureKappaZ(mag), TrueCurvatureKappaP(mag));
}

inline CylindricalVectorLvl0 createTrueCurvatureNablaB( const TokamakMagneticField& mag){
return CylindricalVectorLvl0( TrueCurvatureNablaBR(mag), TrueCurvatureNablaBZ(mag), TrueCurvatureNablaBP(mag));
}

inline CylindricalVectorLvl0 createGradPsip( const TokamakMagneticField& mag){
return CylindricalVectorLvl0( mag.psipR(), mag.psipZ(),Constant(0));
}


struct BHatRR: public aCylindricalFunctor<BHatRR>
{
BHatRR( const TokamakMagneticField& mag): m_invB(mag), m_br(mag), m_mag(mag){}
double do_compute( double R, double Z) const
{
double psipZ = m_mag.psipZ()(R,Z);
double psipRZ = m_mag.psipRZ()(R,Z);
double binv = m_invB(R,Z);
return -psipZ*m_mag.R0()*binv/R/R + psipRZ*binv*m_mag.R0()/R
-psipZ*m_mag.R0()/R*binv*binv*m_br(R,Z);
}
private:
InvB m_invB;
BR m_br;
TokamakMagneticField m_mag;
};
struct BHatRZ: public aCylindricalFunctor<BHatRZ>
{
BHatRZ( const TokamakMagneticField& mag): m_invB(mag), m_bz(mag), m_mag(mag){}
double do_compute( double R, double Z) const
{
double psipZ = m_mag.psipZ()(R,Z);
double psipZZ = m_mag.psipZZ()(R,Z);
double binv = m_invB(R,Z);
return m_mag.R0()/R*( psipZZ*binv  -binv*binv*m_bz(R,Z)*psipZ );
}
private:
InvB m_invB;
BZ m_bz;
TokamakMagneticField m_mag;
};
struct BHatZR: public aCylindricalFunctor<BHatZR>
{
BHatZR( const TokamakMagneticField& mag): m_invB(mag), m_br(mag), m_mag(mag){}
double do_compute( double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z);
double psipRR = m_mag.psipRR()(R,Z);
double binv = m_invB(R,Z);
return +psipR*m_mag.R0()*binv/R/R - psipRR*binv*m_mag.R0()/R
+psipR*m_mag.R0()/R*binv*binv*m_br(R,Z);
}
private:
InvB m_invB;
BR m_br;
TokamakMagneticField m_mag;
};
struct BHatZZ: public aCylindricalFunctor<BHatZZ>
{
BHatZZ( const TokamakMagneticField& mag): m_invB(mag), m_bz(mag), m_mag(mag){}
double do_compute( double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z);
double psipRZ = m_mag.psipRZ()(R,Z);
double binv = m_invB(R,Z);
return -m_mag.R0()/R*( psipRZ*binv  -binv*binv*m_bz(R,Z)*psipR );
}
private:
InvB m_invB;
BZ m_bz;
TokamakMagneticField m_mag;
};
struct BHatPR: public aCylindricalFunctor<BHatPR>
{
BHatPR( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_br(mag){ }
double do_compute( double R, double Z) const
{
double binv = m_invB(R,Z);
double ipol = m_mag.ipol()(R,Z);
double ipolR = m_mag.ipolR()(R,Z);
return -binv*binv*m_br(R,Z)*m_mag.R0()*ipol/R/R
- 2./R/R/R*binv*m_mag.R0()*ipol
+ binv *m_mag.R0()/R/R*ipolR;
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BR m_br;
};
struct BHatPZ: public aCylindricalFunctor<BHatPZ>
{
BHatPZ( const TokamakMagneticField& mag): m_mag(mag), m_invB(mag), m_bz(mag){ }
double do_compute( double R, double Z) const
{
double binv = m_invB(R,Z);
double ipol = m_mag.ipol()(R,Z);
double ipolZ = m_mag.ipolZ()(R,Z);
return -binv*binv*m_bz(R,Z)*m_mag.R0()*ipol/R/R
+ binv *m_mag.R0()/R/R*ipolZ;
}
private:
TokamakMagneticField m_mag;
InvB m_invB;
BZ m_bz;
};

struct DivVVP: public aCylindricalFunctor<DivVVP>
{
DivVVP( const TokamakMagneticField& mag): m_mag(mag),
m_bhatP(mag){ }
double do_compute( double R, double Z) const
{
double ipol = m_mag.ipol()(R,Z), ipolR = m_mag.ipolR()(R,Z),
ipolZ  = m_mag.ipolZ()(R,Z);
double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z);
double bphi = m_bhatP(R,Z);
return -(psipZ*(ipolR/R - 2.*ipol/R/R) - ipolZ/R*psipR)/
(ipol*ipol + psipR*psipR + psipZ*psipZ)/bphi/bphi;
}
private:
TokamakMagneticField m_mag;
BHatP m_bhatP;
};


inline CylindricalVectorLvl1 createBHat( const TokamakMagneticField& mag){
return CylindricalVectorLvl1( BHatR(mag), BHatZ(mag), BHatP(mag),
Divb(mag), DivVVP(mag)
);
}


struct RhoP: public aCylindricalFunctor<RhoP>
{
RhoP( const TokamakMagneticField& mag): m_mag(mag){
double RO = m_mag.R0(), ZO = 0;
try{
findOpoint( mag.get_psip(), RO, ZO);
m_psipmin = m_mag.psip()(RO, ZO);
} catch ( dg::Error& err)
{
m_psipmin = 1.;
if( mag.params().getDescription() == description::centeredX)
m_psipmin = -10;
}
}
double do_compute( double R, double Z) const
{
return sqrt( 1.-m_mag.psip()(R,Z)/m_psipmin ) ;
}
private:
double m_psipmin;
TokamakMagneticField m_mag;

};

struct Hoo : public dg::geo::aCylindricalFunctor<Hoo>
{
Hoo( dg::geo::TokamakMagneticField mag): m_mag(mag){}
double do_compute( double R, double Z) const
{
double psipR = m_mag.psipR()(R,Z), psipZ = m_mag.psipZ()(R,Z), ipol = m_mag.ipol()(R,Z);
double psip2 = psipR*psipR+psipZ*psipZ;
if( psip2 == 0)
psip2 = 1e-16;
return (ipol*ipol + psip2)/R/R/psip2;
}
private:
dg::geo::TokamakMagneticField m_mag;
};

struct WallDirection : public dg::geo::aCylindricalFunctor<WallDirection>
{

WallDirection( dg::geo::TokamakMagneticField mag, std::vector<double>
vertical, std::vector<double> horizontal) : m_vertical(vertical),
m_horizontal(horizontal), m_BR( mag), m_BZ(mag){}

WallDirection( dg::geo::TokamakMagneticField mag,
dg::Grid2d walls) : m_vertical({walls.x0(), walls.x1()}),
m_horizontal({walls.y0(), walls.y1()}), m_BR( mag), m_BZ(mag){}
double do_compute ( double R, double Z) const
{
std::vector<double> v_dist(1,1e100), h_dist(1,1e100);
for( auto v : m_vertical)
v_dist.push_back( R-v );
for( auto h : m_horizontal)
h_dist.push_back( Z-h );
double v_min = *std::min_element( v_dist.begin(), v_dist.end(),
[](double a, double b){ return fabs(a) < fabs(b);} );
double h_min = *std::min_element( h_dist.begin(), h_dist.end(),
[](double a, double b){ return fabs(a) < fabs(b);} );
if( fabs(v_min) < fabs(h_min) ) 
{
double br = m_BR( R,Z);
return v_min*br < 0 ? +1 : -1;
}
else 
{
double bz = m_BZ( R,Z);
return h_min*bz < 0 ? +1 : -1;
}
}
private:
std::vector<double> m_vertical, m_horizontal;
dg::geo::BFieldR m_BR;
dg::geo::BFieldZ m_BZ;
};

} 
} 

