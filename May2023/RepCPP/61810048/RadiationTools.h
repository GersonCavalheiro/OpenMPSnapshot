
#ifndef RADIATIONTOOLS_H
#define RADIATIONTOOLS_H

#include <cmath>
#include "Table.h"

class RadiationTools {

public:


static inline double __attribute__((always_inline)) getHNielFitOrder10(double particle_chi)
{
const double logchi1 = std::log(particle_chi);
const double logchi2 = logchi1 * logchi1;
const double logchi3 = logchi2 * logchi1;
const double logchi4 = logchi3 * logchi1;
const double logchi5 = logchi4 * logchi1;
return std::exp(-3.231764974833856e-08 * logchi5*logchi5
-7.574417415366786e-07 * logchi5*logchi4
-5.437005218419013e-06 * logchi5*logchi3
-4.359062260446135e-06 * logchi5*logchi2
+5.417842511821415e-05 * logchi5*logchi1
-1.263905701127627e-04 * logchi5
+9.899812622393002e-04 * logchi4
+1.076648497464146e-02 * logchi3
-1.624860613422593e-01 * logchi2
+1.496340836237785e+00 * logchi1
-2.756744141581370e+00);
}

static inline double __attribute__((always_inline)) getHNielFitOrder5(double particle_chi)
{

const double logchi1 = std::log(particle_chi);
const double logchi2 = logchi1 * logchi1;
const double logchi3 = logchi2 * logchi1;
return std::exp(+1.399937206900322e-04 * logchi3*logchi2
+3.123718241260330e-03 * logchi3*logchi1
+1.096559086628964e-02 * logchi3
-1.733977278199592e-01 * logchi2
+1.492675770100125e+00 * logchi1
-2.748991631516466e+00 );
}

static inline double __attribute__((always_inline)) getHNielFitRidgers(double particle_chi)
{
const double chi2 = particle_chi * particle_chi;
const double chi3 = chi2 * particle_chi;
return chi3*1.9846415503393384
*std::pow(
1.0 + (1. + 4.528*particle_chi)*std::log(1.+12.29*particle_chi) + 4.632*chi2
,-7./6.
);
}

static inline double __attribute__((always_inline)) computeGRidgers(double particle_chi)
{
return std::pow(1. + 4.8*(1.0+particle_chi)*std::log(1. + 1.7*particle_chi)
+ 2.44*particle_chi*particle_chi,-2./3.);
};

static inline double __attribute__((always_inline)) computeF1Nu(double nu)
{
if (nu<0.1)      return 2.149528241483088*std::pow(nu,-0.6666666666666667) - 1.813799364234217;
else if (nu>10)  return 1.253314137315500*std::pow(nu,-0.5)*exp(-nu);
else {
const double lognu = std::log(nu);
double lognu_power_n = lognu;
double f = -4.341018460806052e-01 - 1.687909081004528e+00 * lognu_power_n;
lognu_power_n *= lognu;
f -= 4.575331390887448e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 1.570476212230771e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 5.349995695960174e-02 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 1.042081355552157e-02 * lognu_power_n; 

return std::exp(f);


}
}

static inline double __attribute__((always_inline)) computeF2Nu(double nu)
{
if (nu<0.05)     return 1.074764120720013*std::pow(nu,-0.6666666666666667);
else if (nu>10)  return 1.253314137315500*std::pow(nu,-0.5)*exp(-nu);
else {
const double lognu = std::log(nu);
double lognu_power_n = lognu;
double f = -7.121012104149862e-01 - 1.539212709860801e+00 * lognu_power_n;
lognu_power_n *= lognu;
f -= 4.589601096726573e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 1.782660550734939e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 5.412029310872778e-02 * lognu_power_n; 

lognu_power_n *= lognu;
f -= 7.694562217592761e-03 * lognu_power_n; 

return std::exp(f);


}
}

static inline double __attribute__((always_inline)) computeBesselPartsRadiatedPower(double nu, double cst)
{
double f1, f2;
if (nu<0.1)
{
f2 = 1.074764120720013 / cbrt(nu*nu);
f1 = 2*f2 - 1.813799364234217;
return f1 + cst*f2;
}
else if (nu>10)
{
return (1.+cst)*1.253314137315500*std::exp(-nu)/std::sqrt(nu);
}
else
{
const double lognu = std::log(nu);
double lognu_power_n = lognu;

f1 = - 4.364684279797524e-01;
f2 = - 7.121012104149862e-01;
f1 -= 1.670543589881836e+00 * lognu_power_n; 
f2 -= 1.539212709860801e+00 * lognu_power_n; 

lognu_power_n *= lognu;
f1 -= 4.533108925728350e-01 * lognu_power_n; 
f2 -= 4.589601096726573e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f1 -= 1.723519212869859e-01 * lognu_power_n; 
f2 -= 1.782660550734939e-01 * lognu_power_n; 

lognu_power_n *= lognu;
f1 -= 5.431864123685266e-02 * lognu_power_n; 
f2 -= 5.412029310872778e-02 * lognu_power_n; 

lognu_power_n *= lognu;
f1 -= 7.892740572869308e-03 * lognu_power_n; 
f2 -= 7.694562217592761e-03 * lognu_power_n; 


return std::exp(f1)+cst*std::exp(f2);
}
}

private:


};
#endif
