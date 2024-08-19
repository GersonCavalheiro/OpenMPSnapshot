
#pragma once

#include "decs.hpp"

#define ind_sph(i, j, k) ( (((k)+n3) % n3) * n2 * n1 + (j) * n1 + (i))
#define ind_periodic(i, j, k) ( (((k)+n3) % n3) * n2 * n1 + (((j)+n2) % n2) * n1 + (((i)+n1) % n1) )




KOKKOS_INLINE_FUNCTION void Xtoijk(const GReal XG[GR_DIM],
const GReal startx[GR_DIM],
const GReal dx[GR_DIM],
int& i, int& j, int& k, GReal del[GR_DIM],
bool nearest=false)
{
GReal phi = XG[3];

if (nearest) {
i = (int) ((XG[1] - startx[1]) / dx[1] + 1000) - 1000;
j = (int) ((XG[2] - startx[2]) / dx[2] + 1000) - 1000;
k = (int) ((phi   - startx[3]) / dx[3] + 1000) - 1000;
} else {
i = (int) ((XG[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
j = (int) ((XG[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
k = (int) ((phi   - startx[3]) / dx[3] - 0.5 + 1000) - 1000;
}

del[1] = (XG[1] - ((i + 0.5) * dx[1] + startx[1])) / dx[1];
del[2] = (XG[2] - ((j + 0.5) * dx[2] + startx[2])) / dx[2];
del[3] = (phi   - ((k + 0.5) * dx[3] + startx[3])) / dx[3];
}

KOKKOS_INLINE_FUNCTION void ijktoX(const GReal startx[GR_DIM], const GReal dx[GR_DIM],
const int& i, const int& j, const int& k,
GReal XG[GR_DIM])
{
XG[0] = 0.;
XG[1] = startx[1] + (i + 0.5) * dx[1];
XG[2] = startx[2] + (j + 0.5) * dx[2];
XG[3] = startx[3] + (k + 0.5) * dx[3];
}


KOKKOS_INLINE_FUNCTION Real linear_interp(const GRCoordinates& G, const GReal X[GR_DIM],
const GReal startx[GR_DIM],
const GReal dx[GR_DIM], const bool& is_spherical, const bool& weight_by_gdet,
const int& n3, const int& n2, const int& n1,
const Real *var)
{
GReal del[GR_DIM];
int i, j, k;
Xtoijk(X, startx, dx, i, j, k, del);

Real interp;
if (is_spherical) {
if (i < 0) { i = 0; del[1] = 0; }
if (i > n1-2) { i = n1 - 2; del[1] = 1; }
if (j < 0) { j = 0; del[2] = 0; }
if (j > n2-2) { j = n2 - 2; del[2] = 1; }

if (weight_by_gdet) {
GReal Xtmp[GR_DIM];
ijktoX(startx, dx, i, j, k, Xtmp);
GReal g_ij = G.coords.gdet_native(Xtmp);
ijktoX(startx, dx, i + 1, j, k, Xtmp);
GReal g_i1j = G.coords.gdet_native(Xtmp);
ijktoX(startx, dx, i, j + 1, k, Xtmp);
GReal g_ij1 = G.coords.gdet_native(Xtmp);
ijktoX(startx, dx, i + 1, j + 1, k, Xtmp);
GReal g_i1j1 = G.coords.gdet_native(Xtmp);

interp = var[ind_sph(i    , j    , k)]*g_ij*(1. - del[1])*(1. - del[2]) +
var[ind_sph(i    , j + 1, k)]*g_ij1*(1. - del[1])*del[2] +
var[ind_sph(i + 1, j    , k)]*g_i1j*del[1]*(1. - del[2]) +
var[ind_sph(i + 1, j + 1, k)]*g_i1j1*del[1]*del[2];

if (n3 > 1) {
interp = (1. - del[3])*interp +
del[3]*(var[ind_sph(i    , j    , k + 1)]*g_ij*(1. - del[1])*(1. - del[2]) +
var[ind_sph(i    , j + 1, k + 1)]*g_ij1*(1. - del[1])*del[2] +
var[ind_sph(i + 1, j    , k + 1)]*g_i1j*del[1]*(1. - del[2]) +
var[ind_sph(i + 1, j + 1, k + 1)]*g_i1j1*del[1]*del[2]);
}
interp /= G.coords.gdet_native(X);
} else {
interp = var[ind_sph(i    , j    , k)]*(1. - del[1])*(1. - del[2]) +
var[ind_sph(i    , j + 1, k)]*(1. - del[1])*del[2] +
var[ind_sph(i + 1, j    , k)]*del[1]*(1. - del[2]) +
var[ind_sph(i + 1, j + 1, k)]*del[1]*del[2];

if (n3 > 1) {
interp = (1. - del[3])*interp +
del[3]*(var[ind_sph(i    , j    , k + 1)]*(1. - del[1])*(1. - del[2]) +
var[ind_sph(i    , j + 1, k + 1)]*(1. - del[1])*del[2] +
var[ind_sph(i + 1, j    , k + 1)]*del[1]*(1. - del[2]) +
var[ind_sph(i + 1, j + 1, k + 1)]*del[1]*del[2]);
}
}
} else {
interp = var[ind_periodic(i    , j    , k)]*(1. - del[1])*(1. - del[2]) +
var[ind_periodic(i    , j + 1, k)]*(1. - del[1])*del[2] +
var[ind_periodic(i + 1, j    , k)]*del[1]*(1. - del[2]) +
var[ind_periodic(i + 1, j + 1, k)]*del[1]*del[2];

if (n3 > 1) {
interp = (1. - del[3])*interp +
del[3]*(var[ind_periodic(i    , j    , k + 1)]*(1. - del[1])*(1. - del[2]) +
var[ind_periodic(i    , j + 1, k + 1)]*(1. - del[1])*del[2] +
var[ind_periodic(i + 1, j    , k + 1)]*del[1]*(1. - del[2]) +
var[ind_periodic(i + 1, j + 1, k + 1)]*del[1]*del[2]);
}
}

return interp;
}

