#include <cmath>
#include <iostream>

#include "3x3-C/dsyevh3.h"

#include "constants.hpp"
#include "parameters.hpp"
#include "matprops.hpp"
#include "rheology.hpp"
#include "utils.hpp"


static void principal_stresses3(const double* s, double p[3], double v[3][3])
{


double a[3][3];
a[0][0] = s[0];
a[1][1] = s[1];
a[2][2] = s[2];
a[0][1] = s[3];
a[0][2] = s[4];
a[1][2] = s[5];

dsyevh3(a, v, p);

if (p[0] > p[1]) {
double tmp, b[3];
tmp = p[0];
p[0] = p[1];
p[1] = tmp;
for (int i=0; i<3; ++i)
b[i] = v[i][0];
for (int i=0; i<3; ++i)
v[i][0] = v[i][1];
for (int i=0; i<3; ++i)
v[i][1] = b[i];
}
if (p[1] > p[2]) {
double tmp, b[3];
tmp = p[1];
p[1] = p[2];
p[2] = tmp;
for (int i=0; i<3; ++i)
b[i] = v[i][1];
for (int i=0; i<3; ++i)
v[i][1] = v[i][2];
for (int i=0; i<3; ++i)
v[i][2] = b[i];
}
if (p[0] > p[1]) {
double tmp, b[3];
tmp = p[0];
p[0] = p[1];
p[1] = tmp;
for (int i=0; i<3; ++i)
b[i] = v[i][0];
for (int i=0; i<3; ++i)
v[i][0] = v[i][1];
for (int i=0; i<3; ++i)
v[i][1] = b[i];
}
}


static void principal_stresses2(const double* s, double p[2],
double& cos2t, double& sin2t)
{


double s0 = 0.5 * (s[0] + s[1]);
double rad = second_invariant(s);

p[0] = s0 - rad;
p[1] = s0 + rad;

{
const double eps = 1e-15;
double a = 0.5 * (s[0] - s[1]);
double b = - rad; 
if (b < -eps) {
cos2t = a / b;
sin2t = s[2] / b;
}
else {
cos2t = 1;
sin2t = 0;
}
}
}


static void elastic(double bulkm, double shearm, const double* de, double* s)
{

double lambda = bulkm - 2. /3 * shearm;
double dev = trace(de);

for (int i=0; i<NDIMS; ++i)
s[i] += 2 * shearm * de[i] + lambda * dev;
for (int i=NDIMS; i<NSTR; ++i)
s[i] += 2 * shearm * de[i];
}


static void maxwell(double bulkm, double shearm, double viscosity, double dt,
double dv, const double* de, double* s)
{
double tmp = 0.5 * dt * shearm / viscosity;
double f1 = 1 - tmp;
double f2 = 1 / (1  + tmp);

double dev = trace(de) / NDIMS;
double s0 = trace(s) / NDIMS;

for (int i=0; i<NDIMS; ++i)
s[i] = ((s[i] - s0) * f1 + 2 * shearm * (de[i] - dev)) * f2 + s0 + bulkm * dv;
for (int i=NDIMS; i<NSTR; ++i)
s[i] = (s[i] * f1 + 2 * shearm * de[i]) * f2;
}


static void viscous(double bulkm, double viscosity, double total_dv,
const double* edot, double* s)
{


double dev = trace(edot) / NDIMS;

for (int i=0; i<NDIMS; ++i)
s[i] = 2 * viscosity * (edot[i] - dev) + bulkm * total_dv;
for (int i=NDIMS; i<NSTR; ++i)
s[i] = 2 * viscosity * edot[i];
}


static void elasto_plastic(double bulkm, double shearm,
double amc, double anphi, double anpsi,
double hardn, double ten_max,
const double* de, double& depls, double* s,
int &failure_mode)
{


elastic(bulkm, shearm, de, s);
depls = 0;
failure_mode = 0;

double p[NDIMS];
#ifdef THREED
double v[3][3];
principal_stresses3(s, p, v);
#else
double cos2t, sin2t;
principal_stresses2(s, p, cos2t, sin2t);
#endif

double fs = p[0] - p[NDIMS-1] * anphi + amc;
double ft = p[NDIMS-1] - ten_max;

if (fs > 0 && ft < 0) {
return;
}

double pa = std::sqrt(1 + anphi*anphi) + anphi;
double ps = ten_max * anphi - amc;
double h = p[NDIMS-1] - ten_max + pa * (p[0] - ps);
double a1 = bulkm + 4. / 3 * shearm;
double a2 = bulkm - 2. / 3 * shearm;

double alam;
if (h < 0) {
failure_mode = 10;

alam = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + 2*std::sqrt(anphi)*hardn);
p[0] -= alam * (a1 - a2 * anpsi);
#ifdef THREED
p[1] -= alam * (a2 - a2 * anpsi);
#endif
p[NDIMS-1] -= alam * (a2 - a1 * anpsi);

#ifdef THREED

depls = std::fabs(alam) * std::sqrt((7 + 4*anpsi + 7*anpsi*anpsi) / 18);
#else

depls = std::fabs(alam) * std::sqrt((3 + 2*anpsi + 3*anpsi*anpsi) / 8);
#endif
}
else {
failure_mode = 1;

alam = ft / a1;
p[0] -= alam * a2;
#ifdef THREED
p[1] -= alam * a2;
#endif
p[NDIMS-1] -= alam * a1;

#ifdef THREED

depls = std::fabs(alam) * std::sqrt(7. / 18);
#else

depls = std::fabs(alam) * std::sqrt(3. / 8);
#endif
}

{
#ifdef THREED
double ss[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
for(int m=0; m<3; m++) {
for(int n=m; n<3; n++) {
for(int k=0; k<3; k++) {
ss[m][n] += v[m][k] * v[n][k] * p[k];
}
}
}
s[0] = ss[0][0];
s[1] = ss[1][1];
s[2] = ss[2][2];
s[3] = ss[0][1];
s[4] = ss[0][2];
s[5] = ss[1][2];
#else
double dc2 = (p[0] - p[1]) * cos2t;
double dss = p[0] + p[1];
s[0] = 0.5 * (dss + dc2);
s[1] = 0.5 * (dss - dc2);
s[2] = 0.5 * (p[0] - p[1]) * sin2t;
#endif
}
}


static void elasto_plastic2d(double bulkm, double shearm,
double amc, double anphi, double anpsi,
double hardn, double ten_max,
const double* de, double& depls,
double* s, double &syy,
int &failure_mode)
{




depls = 0;
failure_mode = 0;

double a1 = bulkm + 4. / 3 * shearm;
double a2 = bulkm - 2. / 3 * shearm;
double sxx = s[0] + de[1]*a2 + de[0]*a1;
double szz = s[1] + de[0]*a2 + de[1]*a1;
double sxz = s[2] + de[2]*2*shearm;
syy += (de[0] + de[1]) * a2; 


double p[3];

double cos2t, sin2t;
int n1, n2, n3;

{
double s0 = 0.5 * (sxx + szz);
double rad = 0.5 * std::sqrt((sxx-szz)*(sxx-szz) + 4*sxz*sxz);

double si = s0 - rad;
double sii = s0 + rad;

const double eps = 1e-15;
if (rad > eps) {
cos2t = 0.5 * (szz - sxx) / rad;
sin2t = -sxz / rad;
}
else {
cos2t = 1;
sin2t = 0;
}

#if 1
if (syy > sii) {
n1 = 0;
n2 = 1;
n3 = 2;
p[0] = si;
p[1] = sii;
p[2] = syy;
}
else if (syy < si) {
n1 = 1;
n2 = 2;
n3 = 0;
p[0] = syy;
p[1] = si;
p[2] = sii;
}
else {
n1 = 0;
n2 = 2;
n3 = 1;
p[0] = si;
p[1] = syy;
p[2] = sii;
}
#else


n1 = 0;
n2 = 2;
p[0] = si;
p[1] = syy;
p[2] = sii;
#endif
}

if( p[0] >= ten_max ) {
s[0] = s[1] = syy = ten_max;
s[2] = 0.0;
failure_mode = 1;
return;
}

if( p[1] >= ten_max ) {
p[1] = p[2] = ten_max;
failure_mode = 2;
}

else if( p[2] >= ten_max ) {
p[2] = ten_max;
failure_mode = 3;
}


double fs = p[0] - p[2] * anphi + amc;
if (fs >= 0.0) {
s[0] = sxx;
s[1] = szz;
s[2] = sxz;
return;
}

failure_mode += 10;

const double alams = fs / (a1 - a2*anpsi + a1*anphi*anpsi - a2*anphi + hardn);
p[0] -= alams * (a1 - a2 * anpsi);
p[1] -= alams * (a2 - a2 * anpsi);
p[2] -= alams * (a2 - a1 * anpsi);

depls = 0.5 * std::fabs(alams + alams * anpsi);

if( p[0] >= ten_max ) {
s[0] = s[1] = syy = ten_max;
s[2] = 0.0;
failure_mode += 20;
return;
}

if( p[1] >= ten_max ) {
p[1] = p[2] = ten_max;
failure_mode += 20;
}

else if( p[2] >= ten_max ) {
p[2] = ten_max;
failure_mode += 20;
}


{
double dc2 = (p[n1] - p[n2]) * cos2t;
double dss = p[n1] + p[n2];
s[0] = 0.5 * (dss + dc2);
s[1] = 0.5 * (dss - dc2);
s[2] = 0.5 * (p[n1] - p[n2]) * sin2t;
syy = p[n3];
}
}


void update_stress(const Variables& var, tensor_t& stress,
double_vec& stressyy, double_vec& dpressure,
tensor_t& strain, double_vec& plstrain,
double_vec& delta_plstrain, tensor_t& strain_rate)
{
const int rheol_type = var.mat->rheol_type;

#pragma omp parallel for default(none)                           \
shared(var, stress, stressyy, dpressure, strain, plstrain, delta_plstrain, \
strain_rate, rheol_type, std::cerr)
for (int e=0; e<var.nelem; ++e) {
double* s = stress[e];
double& syy = stressyy[e];
double* es = strain[e];
double* edot = strain_rate[e];
double old_s = trace(s);

if(1){
double div = trace(edot);
for (int i=0; i<NDIMS; ++i) {
edot[i] += ((*var.edvoldt)[e] - div) / NDIMS;  
}
}

for (int i=0; i<NSTR; ++i) {
es[i] += edot[i] * var.dt;
}

double de[NSTR];
for (int i=0; i<NSTR; ++i) {
de[i] = edot[i] * var.dt;
}

switch (rheol_type) {
case MatProps::rh_elastic:
{
double bulkm = var.mat->bulkm(e);
double shearm = var.mat->shearm(e);
elastic(bulkm, shearm, de, s);
}
break;
case MatProps::rh_viscous:
{
double bulkm = var.mat->bulkm(e);
double viscosity = var.mat->visc(e);
double total_dv = trace(es);
viscous(bulkm, viscosity, total_dv, edot, s);
}
break;
case MatProps::rh_maxwell:
{
double bulkm = var.mat->bulkm(e);
double shearm = var.mat->shearm(e);
double viscosity = var.mat->visc(e);
double dv = (*var.volume)[e] / (*var.volume_old)[e] - 1;
maxwell(bulkm, shearm, viscosity, var.dt, dv, de, s);
}
break;
case MatProps::rh_ep:
{
double depls = 0;
double bulkm = var.mat->bulkm(e);
double shearm = var.mat->shearm(e);
double amc, anphi, anpsi, hardn, ten_max;
var.mat->plastic_props(e, plstrain[e],
amc, anphi, anpsi, hardn, ten_max);
int failure_mode;
if (var.mat->is_plane_strain) {
elasto_plastic2d(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
de, depls, s, syy, failure_mode);
}
else {
elasto_plastic(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
de, depls, s, failure_mode);
}
plstrain[e] += depls;
delta_plstrain[e] = depls;
}
break;
case MatProps::rh_evp:
{
double depls = 0;
double bulkm = var.mat->bulkm(e);
double shearm = var.mat->shearm(e);
double viscosity = var.mat->visc(e);
double dv = (*var.volume)[e] / (*var.volume_old)[e] - 1;
double sv[NSTR];
for (int i=0; i<NSTR; ++i) sv[i] = s[i];
maxwell(bulkm, shearm, viscosity, var.dt, dv, de, sv);
double svII = second_invariant2(sv);

double amc, anphi, anpsi, hardn, ten_max;
var.mat->plastic_props(e, plstrain[e],
amc, anphi, anpsi, hardn, ten_max);
double sp[NSTR], spyy;
for (int i=0; i<NSTR; ++i) sp[i] = s[i];
int failure_mode;
if (var.mat->is_plane_strain) {
spyy = syy;
elasto_plastic2d(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
de, depls, sp, spyy, failure_mode);
}
else {
elasto_plastic(bulkm, shearm, amc, anphi, anpsi, hardn, ten_max,
de, depls, sp, failure_mode);
}
double spII = second_invariant2(sp);

if (svII < spII)
for (int i=0; i<NSTR; ++i) s[i] = sv[i];
else {
for (int i=0; i<NSTR; ++i) s[i] = sp[i];
plstrain[e] += depls;
delta_plstrain[e] = depls;
syy = spyy;
}
}
break;
default:
std::cerr << "Error: unknown rheology type: " << rheol_type << "\n";
std::exit(1);
break;
}
dpressure[e] = trace(s) - old_s;
}
}
