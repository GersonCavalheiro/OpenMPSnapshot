#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "dg/algorithm.h"
#include "modified.h"
#include "polynomial_parameters.h"
#include "magnetic_field.h"



namespace dg
{
namespace geo
{

namespace polynomial
{


struct Psip: public aCylindricalFunctor<Psip>
{

Psip( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp), m_horner(gp.c, gp.M, gp.N) {}
double do_compute(double R, double Z) const
{
return m_R0*m_pp*m_horner( R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};

struct PsipR: public aCylindricalFunctor<PsipR>
{
PsipR( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
std::vector<double>  beta ( (gp.M-1)*gp.N);
for( unsigned i=0; i<gp.M-1; i++)
for( unsigned j=0; j<gp.N; j++)
beta[i*gp.N+j] = (double)(i+1)*gp.c[ ( i+1)*gp.N +j];
m_horner = Horner2d( beta, gp.M-1, gp.N);
}
double do_compute(double R, double Z) const
{
return m_pp*m_horner( R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};
struct PsipRR: public aCylindricalFunctor<PsipRR>
{
PsipRR( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
std::vector<double>  beta ( (gp.M-2)*gp.N);
for( unsigned i=0; i<gp.M-2; i++)
for( unsigned j=0; j<gp.N; j++)
beta[i*gp.N+j] = (double)((i+2)*(i+1))*gp.c[ (i+2)*gp.N +j];
m_horner = Horner2d( beta, gp.M-2, gp.N);
}
double do_compute(double R, double Z) const
{
return m_pp/m_R0*m_horner( R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};
struct PsipZ: public aCylindricalFunctor<PsipZ>
{
PsipZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
std::vector<double>  beta ( gp.M*(gp.N-1));
for( unsigned i=0; i<gp.M; i++)
for( unsigned j=0; j<gp.N-1; j++)
beta[i*(gp.N-1)+j] = (double)(j+1)*gp.c[ i*gp.N +j+1];
m_horner = Horner2d( beta, gp.M, gp.N-1);
}
double do_compute(double R, double Z) const
{
return m_pp*m_horner( R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};
struct PsipZZ: public aCylindricalFunctor<PsipZZ>
{
PsipZZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
std::vector<double>  beta ( gp.M*(gp.N-2));
for( unsigned i=0; i<gp.M; i++)
for( unsigned j=0; j<gp.N-2; j++)
beta[i*(gp.N-2)+j] = (double)((j+2)*(j+1))*gp.c[ i*gp.N +j+2];
m_horner = Horner2d( beta, gp.M, gp.N-2);
}
double do_compute(double R, double Z) const
{
return m_pp/m_R0*m_horner(R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};
struct PsipRZ: public aCylindricalFunctor<PsipRZ>
{
PsipRZ( Parameters gp ): m_R0(gp.R_0),  m_pp(gp.pp){
std::vector<double>  beta ( (gp.M-1)*(gp.N-1));
for( unsigned i=0; i<gp.M-1; i++)
for( unsigned j=0; j<gp.N-1; j++)
beta[i*(gp.N-1)+j] = (double)((j+1)*(i+1))*gp.c[ (i+1)*gp.N +j+1];
m_horner = Horner2d( beta, gp.M-1, gp.N-1);
}
double do_compute(double R, double Z) const
{
return m_pp/m_R0*m_horner(R/m_R0,Z/m_R0);
}
private:
double m_R0, m_pp;
Horner2d m_horner;
};

static inline dg::geo::CylindricalFunctorsLvl2 createPsip( Parameters gp)
{
return CylindricalFunctorsLvl2( Psip(gp), PsipR(gp), PsipZ(gp),
PsipRR(gp), PsipRZ(gp), PsipZZ(gp));
}
static inline dg::geo::CylindricalFunctorsLvl1 createIpol( Parameters gp)
{
return CylindricalFunctorsLvl1( Constant( gp.pi), Constant(0), Constant(0));
}


} 


static inline dg::geo::TokamakMagneticField createPolynomialField(
dg::geo::polynomial::Parameters gp)
{
MagneticFieldParameters params( gp.a, gp.elongation, gp.triangularity,
equilibrium::polynomial, modifier::none, str2description.at( gp.description));
return TokamakMagneticField( gp.R_0, polynomial::createPsip(gp),
polynomial::createIpol(gp), params);
}

} 
} 

