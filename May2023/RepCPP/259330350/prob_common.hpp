
#pragma once

#include "decs.hpp"


KOKKOS_INLINE_FUNCTION void rotate_polar(const GReal Xin[GR_DIM], const GReal angle, GReal Xout[GR_DIM], const bool spherical=true)
{
if (m::abs(angle) < 1e-20) {
DLOOP1 Xout[mu] = Xin[mu];
return;
}


GReal Xin_cart[GR_DIM] = {0};
if (spherical) {
Xin_cart[1] = Xin[1]*sin(Xin[2])*cos(Xin[3]);
Xin_cart[2] = Xin[1]*sin(Xin[2])*sin(Xin[3]);
Xin_cart[3] = Xin[1]*cos(Xin[2]);
} else {
DLOOP1 Xin_cart[mu] = Xin[mu];
}

GReal R[GR_DIM][GR_DIM] = {0};
R[0][0] = 1;
R[1][1] =  cos(angle);
R[1][3] =  sin(angle);
R[2][2] =  1;
R[3][1] = -sin(angle);
R[3][3] =  cos(angle);

GReal Xout_cart[GR_DIM] = {0};
DLOOP2 Xout_cart[mu] += R[mu][nu] * Xin_cart[nu];

if (spherical) {
Xout[0] = Xin[0];
Xout[1] = Xin[1]; 
Xout[2] = acos(Xout_cart[3]/Xout[1]);
if (m::isnan(Xout[2])) { 
if (Xout_cart[3]/Xout[1] < 0)
Xout[2] = M_PI;
else
Xout[2] = 0.0;
}
Xout[3] = atan2(Xout_cart[2], Xout_cart[1]);
} else {
DLOOP1 Xout[mu] = Xout_cart[mu];
}
}


KOKKOS_INLINE_FUNCTION void set_dXdx_sph2cart(const GReal X[GR_DIM], GReal dXdx[GR_DIM][GR_DIM])
{
const GReal r = X[1], th = X[2], phi = X[3];
dXdx[0][0] = 1;
dXdx[1][1] = sin(th)*cos(phi);
dXdx[1][2] = r*cos(th)*cos(phi);
dXdx[1][3] = -r*sin(th)*sin(phi);
dXdx[2][1] = sin(th)*sin(phi);
dXdx[2][2] = r*cos(th)*sin(phi);
dXdx[2][3] = r*sin(th)*cos(phi);
dXdx[3][1] = cos(th);
dXdx[3][2] = -r*sin(th);
dXdx[3][3] = 0;
}


KOKKOS_INLINE_FUNCTION void rotate_polar_vec(const GReal Xin[GR_DIM], const GReal vin[GR_DIM], const GReal angle,
const GReal Xout[GR_DIM], GReal vout[GR_DIM],
const bool spherical=true)
{
if (m::abs(angle) < 1e-20) {
DLOOP1 vout[mu] = vin[mu];
return;
}


GReal vin_cart[GR_DIM] = {0};
if (spherical) {
GReal dXdx[GR_DIM][GR_DIM] = {0};
set_dXdx_sph2cart(Xin, dXdx);
DLOOP2 vin_cart[mu] += dXdx[mu][nu]*vin[nu];
} else {
DLOOP1 vin_cart[mu] = vin[mu];
}

GReal R[GR_DIM][GR_DIM] = {0};
R[0][0] = 1;
R[1][1] = cos(angle);
R[1][3] = sin(angle);
R[2][2] = 1;
R[3][1] = -sin(angle);
R[3][3] = cos(angle);

GReal vout_cart[GR_DIM] = {0};
DLOOP2 vout_cart[mu] += R[mu][nu] * vin_cart[nu];

if (spherical) {
GReal dXdx[GR_DIM][GR_DIM] = {0}, dxdX[GR_DIM][GR_DIM] = {0};
set_dXdx_sph2cart(Xout, dXdx);
invert(&dXdx[0][0], &dxdX[0][0]);
DLOOP1 vout[mu] = 0;
DLOOP2 vout[mu] += dxdX[mu][nu]*vout_cart[nu];
} else {
DLOOP1 vout[mu] = vout_cart[mu];
}
}


KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[GR_DIM][GR_DIM], Real ucon[GR_DIM])
{
Real AA, BB, CC;

AA = gcov[0][0];
BB = 2. * (gcov[0][1] * ucon[1] +
gcov[0][2] * ucon[2] +
gcov[0][3] * ucon[3]);
CC = 1. + gcov[1][1] * ucon[1] * ucon[1] +
gcov[2][2] * ucon[2] * ucon[2] +
gcov[3][3] * ucon[3] * ucon[3] +
2. * (gcov[1][2] * ucon[1] * ucon[2] +
gcov[1][3] * ucon[1] * ucon[3] +
gcov[2][3] * ucon[2] * ucon[3]);

Real discr = BB * BB - 4. * AA * CC;
ucon[0] = (-BB - m::sqrt(discr)) / (2. * AA);
}


KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[GR_DIM][GR_DIM], const Real ucon[GR_DIM], Real u_prim[NVEC])
{
Real alpha2 = -1.0 / gcon[0][0];
u_prim[0] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
u_prim[1] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
u_prim[2] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}
