#include <omp.h>
#include <iostream>
#include <cmath>
#include <iterator>
#include <fstream>
#include <openacc.h>
#include <typeinfo>

using namespace std;

#if OPENACCCOMPILE == 1
#define OPENACC 1
#else
#define OPENACC 0
#endif

#if FLOATS
#define real float
#else
#define real double
#endif

int main() {
if (OPENACC == 1) {
cout << endl << "OpenACC acceleration enabled from cmake, code should run on GPU." << endl;
} else {
cout << endl << "OpenACC acceleration not enabled from cmake, code should run on CPU." << endl;
}
cout << "Code compiled with " << typeid(real).name() << " (d for double, accurate, f for float, fast)" << endl << flush;

constexpr static real coeff1 = real(9.0 / 8.0);
constexpr static real coeff2 = real(1.0 / 24.0);

const uint nt = 250;
const uint nx = 4096;
const uint nz = 1024;

real dx = 100;
real dz = 100;
real dt = 0.03;

uint ixSource = 125;
uint izSource = 3;
real alpha = 0.01;
real t0 = 2;

real poissons = 0.25;
real vratio = sqrt((1 - 2 * poissons) / (2 * (1 - poissons)));
static real rho[nx][nz] = {{2500}}; 
static real vp[nx][nz] = {{2000}};
static real vs[nx][nz] = {{float(vp[0][0] * vratio)}};
static real taper[nx][nz] = {{1}};

static real t[nt];
static real stf[nt];

for (unsigned int it = 0; it < nt; ++it) {
t[it] = it * dt;
stf[it] = real(exp(-alpha * pow(it - t0 / dt, 2)));
}

real moment[2][2];
moment[0][0] = 1;
moment[0][1] = 0;
moment[1][0] = 0;
moment[1][1] = -1;

static real vx[nx][nz] = {{0}}; 
static real vz[nx][nz] = {{0}}; 
static real txx[nx][nz] = {{0}};
static real tzz[nx][nz] = {{0}};
static real txz[nx][nz] = {{0}};

static real lm[nx][nz] = {{1}};
static real la[nx][nz] = {{1}};
static real mu[nx][nz] = {{1}};
static real b_vx[nx][nz] = {{1}};
static real b_vz[nx][nz] = {{1}};

#pragma omp parallel for collapse(2)
for (uint ix = 0; ix < nx; ++ix) {
for (uint iz = 0; iz < nz; ++iz) {
rho[ix][iz] = rho[0][0];
vp[ix][iz] = vp[0][0];
vs[ix][iz] = vs[0][0];
taper[ix][iz] = taper[0][0];

mu[ix][iz] = real(pow(vs[ix][iz], 2) * rho[ix][iz]);
lm[ix][iz] = real(pow(vp[ix][iz], 2) * rho[ix][iz]);
la[ix][iz] = lm[ix][iz] - 2 * mu[ix][iz];
b_vx[ix][iz] = real(1.0 / rho[ix][iz]);
b_vz[ix][iz] = b_vx[ix][iz];
}
}



real startTime = real(omp_get_wtime());
#pragma acc data copyin( lm, taper, la, mu, b_vx, b_vz, stf, moment) copyout(txx, vz, txz, tzz, vx)
{
for (uint it = 0; it < nt; ++it) {
#pragma acc region
{
#pragma omp parallel for collapse(2)
#pragma acc loop tile(32, 32)
for (uint ix = 2; ix < nx - 2; ++ix) {
for (uint iz = 2; iz < nz - 2; ++iz) {
txx[ix][iz] = taper[ix][iz] *
(txx[ix][iz] +
dt *
(lm[ix][iz] * (
coeff1 * (vx[ix + 1][iz] - vx[ix][iz]) +
coeff2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
la[ix][iz] * (
coeff1 * (vz[ix][iz] - vz[ix][iz - 1]) +
coeff2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
tzz[ix][iz] = taper[ix][iz] *
(tzz[ix][iz] +
dt *
(la[ix][iz] * (
coeff1 * (vx[ix + 1][iz] - vx[ix][iz]) +
coeff2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
(lm[ix][iz]) * (
coeff1 * (vz[ix][iz] - vz[ix][iz - 1]) +
coeff2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
txz[ix][iz] = taper[ix][iz] *
(txz[ix][iz] + dt * mu[ix][iz] * (
(coeff1 * (vx[ix][iz + 1] - vx[ix][iz]) +
coeff2 * (vx[ix][iz - 1] - vx[ix][iz + 2])) / dz +
(coeff1 * (vz[ix][iz] - vz[ix - 1][iz]) +
coeff2 * (vz[ix - 2][iz] - vz[ix + 1][iz])) / dx));

}
}
#pragma omp parallel for collapse(2)
#pragma acc loop tile(32, 32)
for (uint ix = 2; ix < nx - 2; ++ix) {
for (uint iz = 2; iz < nz - 2; ++iz) {
vx[ix][iz] =
taper[ix][iz] *
(vx[ix][iz]
+ b_vx[ix][iz] * dt * (
(coeff1 * (txx[ix][iz] - txx[ix - 1][iz]) +
coeff2 * (txx[ix - 2][iz] - txx[ix + 1][iz])) / dx +
(coeff1 * (txz[ix][iz] - txz[ix][iz - 1]) +
coeff2 * (txz[ix][iz - 2] - txz[ix][iz + 1])) / dz));
vz[ix][iz] =
taper[ix][iz] *
(vz[ix][iz]
+ b_vz[ix][iz] * dt * (
(coeff1 * (txz[ix + 1][iz] - txz[ix][iz]) +
coeff2 * (txz[ix - 1][iz] - txz[ix + 2][iz])) / dx +
(coeff1 * (tzz[ix][iz + 1] - tzz[ix][iz]) +
coeff2 * (tzz[ix][iz - 1] - tzz[ix][iz + 2])) / dz));
}
}

vx[ixSource - 1][izSource] -= moment[0][0] * stf[it] * dt * b_vz[ixSource - 1][izSource] / (dx * dx * dx * dx);
vx[ixSource][izSource] += moment[0][0] * stf[it] * dt * b_vz[ixSource][izSource] / (dx * dx * dx * dx);

vz[ixSource][izSource - 1] -= moment[1][1] * stf[it] * dt * b_vz[ixSource][izSource - 1] / (dz * dz * dz * dz);
vz[ixSource][izSource] += moment[1][1] * stf[it] * dt * b_vz[ixSource][izSource] / (dz * dz * dz * dz);

vx[ixSource - 1][izSource + 1] += 0.25 * moment[0][1] * stf[it] * dt * b_vz[ixSource - 1][izSource + 1] / (dx * dx * dx * dx);
vx[ixSource][izSource + 1] += 0.25 * moment[0][1] * stf[it] * dt * b_vz[ixSource][izSource + 1] / (dx * dx * dx * dx);
vx[ixSource - 1][izSource - 1] -= 0.25 * moment[0][1] * stf[it] * dt * b_vz[ixSource - 1][izSource - 1] / (dx * dx * dx * dx);
vx[ixSource][izSource - 1] -= 0.25 * moment[0][1] * stf[it] * dt * b_vz[ixSource][izSource - 1] / (dx * dx * dx * dx);

vz[ixSource + 1][izSource - 1] += 0.25 * moment[1][0] * stf[it] * dt * b_vz[ixSource + 1][izSource - 1] / (dz * dz * dz * dz);
vz[ixSource + 1][izSource] += 0.25 * moment[1][0] * stf[it] * dt * b_vz[ixSource + 1][izSource] / (dz * dz * dz * dz);
vz[ixSource - 1][izSource - 1] -= 0.25 * moment[1][0] * stf[it] * dt * b_vz[ixSource - 1][izSource - 1] / (dz * dz * dz * dz);
vz[ixSource - 1][izSource] -= 0.25 * moment[1][0] * stf[it] * dt * b_vz[ixSource - 1][izSource] / (dz * dz * dz * dz);
}
}
}

real stopTime = real(omp_get_wtime());
real secsElapsed = stopTime - startTime;
cout << "Seconds elapsed for wave simulation: " << secsElapsed << endl;

real a = 0;
for (const auto &item : vx) {
for (const auto &item2 : item) {
a += item2;
}
}
cout << a << endl;

return 0;
}

