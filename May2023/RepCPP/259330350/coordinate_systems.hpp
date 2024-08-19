
#pragma once

#include <mpark/variant.hpp>

#include "decs.hpp"

#include "matrix.hpp"
#include "kharma_utils.hpp"
#include "root_find.hpp"

#define LEGACY_TH 1






class CartMinkowskiCoords {
public:
const bool spherical = false;
KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
{
DLOOP2 gcov[mu][nu] = (mu == nu) - 2*(mu == 0 && nu == 0);
}
};


class SphMinkowskiCoords {
public:
const bool spherical = true;
KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
{
const GReal r = m::max(Xembed[1], SMALL);
const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
const GReal sth = sin(th);

gzero2(gcov);
gcov[0][0] = 1.;
gcov[1][1] = 1.;
gcov[2][2] = r*r;
gcov[3][3] = m::pow(sth*r, 2);
}
};


class SphKSCoords {
public:
const GReal a;
const bool spherical = true;

KOKKOS_FUNCTION SphKSCoords(GReal spin): a(spin) {};

KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
{
const GReal r = Xembed[1];
const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);

const GReal cos2 = m::pow(cos(th), 2);
const GReal sin2 = m::pow(sin(th), 2);
const GReal rho2 = r*r + a*a*cos2;

gcov[0][0] = -1. + 2.*r/rho2;
gcov[0][1] = 2.*r/rho2;
gcov[0][2] = 0.;
gcov[0][3] = -2.*a*r*sin2/rho2;

gcov[1][0] = 2.*r/rho2;
gcov[1][1] = 1. + 2.*r/rho2;
gcov[1][2] = 0.;
gcov[1][3] = -a*sin2*(1. + 2.*r/rho2);

gcov[2][0] = 0.;
gcov[2][1] = 0.;
gcov[2][2] = rho2;
gcov[2][3] = 0.;

gcov[3][0] = -2.*a*r*sin2/rho2;
gcov[3][1] = -a*sin2*(1. + 2.*r/rho2);
gcov[3][2] = 0.;
gcov[3][3] = sin2*(rho2 + a*a*sin2*(1. + 2.*r/rho2));
}

KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
{
GReal r = Xembed[1];
Real trans[GR_DIM][GR_DIM];
DLOOP2 trans[mu][nu] = (mu == nu);
trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
trans[3][1] = a/(r*r - 2.*r + a*a);

gzero(vcon);
DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
}

KOKKOS_INLINE_FUNCTION void vec_to_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
{
GReal r = Xembed[1];
GReal rtrans[GR_DIM][GR_DIM], trans[GR_DIM][GR_DIM];
DLOOP2 rtrans[mu][nu] = (mu == nu);
rtrans[0][1] = 2.*r/(r*r - 2.*r + a*a);
rtrans[3][1] = a/(r*r - 2.*r + a*a);
invert(&rtrans[0][0], &trans[0][0]);

gzero(vcon);
DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
}

KOKKOS_INLINE_FUNCTION GReal rhor() const
{
return (1. + m::sqrt(1. - a*a));
}
};


class SphBLCoords {
public:
const GReal a;
const bool spherical = true;

KOKKOS_FUNCTION SphBLCoords(GReal spin): a(spin) {}

KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
{
const GReal r = Xembed[1];
const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
const GReal cth = cos(th), sth = sin(th);

const GReal s2 = sth*sth;
const GReal a2 = a*a;
const GReal r2 = r*r;
const GReal mmu = 1. + a2*cth*cth/r2; 

gzero2(gcov);
gcov[0][0]  = -(1. - 2./(r*mmu));
gcov[0][3]  = -2.*a*s2/(r*mmu);
gcov[1][1]   = mmu/(1. - 2./r + a2/r2);
gcov[2][2]   = r2*mmu;
gcov[3][0]  = -2.*a*s2/(r*mmu);
gcov[3][3]   = s2*(r2 + a2 + 2.*a2*s2/(r*mmu));
}


KOKKOS_INLINE_FUNCTION GReal rhor() const
{
return (1. + m::sqrt(1. - a*a));
}
};




class NullTransform {
public:
KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
{
DLOOP1 Xembed[mu] = Xnative[mu];
}
KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
{
DLOOP1 Xnative[mu] = Xembed[mu];
}
KOKKOS_INLINE_FUNCTION void dxdX(const GReal X[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
{
DLOOP2 dxdX[mu][nu] = (mu == nu);
}
KOKKOS_INLINE_FUNCTION void dXdx(const GReal X[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
{
DLOOP2 dXdx[mu][nu] = (mu == nu);
}
};


class ExponentialTransform {
public:
KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
{
Xembed[0] = Xnative[0];
Xembed[1] = exp(Xnative[1]);
#if LEGACY_TH
Xembed[2] = excise(excise(Xnative[2], 0.0, SMALL), M_PI, SMALL);
#else
Xembed[2] = Xnative[2];
#endif
Xembed[3] = Xnative[3];
}
KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
{
Xnative[0] = Xembed[0];
Xnative[1] = log(Xembed[1]);
Xnative[2] = Xembed[2];
Xnative[3] = Xembed[3];
}

KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
{
gzero2(dxdX);
dxdX[0][0] = 1.;
dxdX[1][1] = exp(Xnative[1]);
dxdX[2][2] = 1.;
dxdX[3][3] = 1.;
}

KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
{
gzero2(dXdx);
dXdx[0][0] = 1.;
dXdx[1][1] = 1 / exp(Xnative[1]);
dXdx[2][2] = 1.;
dXdx[3][3] = 1.;
}
};


class ModifyTransform {
public:
const GReal hslope;

KOKKOS_FUNCTION ModifyTransform(GReal hslope_in): hslope(hslope_in) {}

KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
{
Xembed[0] = Xnative[0];
Xembed[1] = exp(Xnative[1]);
#if LEGACY_TH
const GReal th = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
Xembed[2] = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
#endif
Xembed[3] = Xnative[3];
}
KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
{
Xnative[0] = Xembed[0];
Xnative[1] = log(Xembed[1]);
Xnative[3] = Xembed[3];
ROOT_FIND
}

KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
{
gzero2(dxdX);
dxdX[0][0] = 1.;
dxdX[1][1] = exp(Xnative[1]);
dxdX[2][2] = M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*Xnative[2]);
dxdX[3][3] = 1.;
}

KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
{
gzero2(dXdx);
dXdx[0][0] = 1.;
dXdx[1][1] = 1 / exp(Xnative[1]);
dXdx[2][2] = 1 / (M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*Xnative[2]));
dXdx[3][3] = 1.;
}
};


class FunkyTransform {
public:
const GReal startx1;
const GReal hslope, poly_xt, poly_alpha, mks_smooth;
GReal poly_norm; 

KOKKOS_FUNCTION FunkyTransform(GReal startx1_in, GReal hslope_in, GReal mks_smooth_in, GReal poly_xt_in, GReal poly_alpha_in):
startx1(startx1_in), hslope(hslope_in), mks_smooth(mks_smooth_in), poly_xt(poly_xt_in), poly_alpha(poly_alpha_in)
{
poly_norm = 0.5 * M_PI * 1./(1. + 1./(poly_alpha + 1.) * 1./m::pow(poly_xt, poly_alpha));
}

KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
{
Xembed[0] = Xnative[0];
Xembed[1] = exp(Xnative[1]);

const GReal thG = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
const GReal y = 2*Xnative[2] - 1.;
const GReal thJ = poly_norm * y * (1. + m::pow(y/poly_xt,poly_alpha) / (poly_alpha + 1.)) + 0.5 * M_PI;
#if LEGACY_TH
const GReal th = thG + exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
Xembed[2] = thG + exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
#endif
Xembed[3] = Xnative[3];
}
KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
{
Xnative[0] = Xembed[0];
Xnative[1] = log(Xembed[1]);
Xnative[3] = Xembed[3];
ROOT_FIND
}

KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
{
gzero2(dxdX);
dxdX[0][0] = 1.;
dxdX[1][1] = exp(Xnative[1]);
dxdX[2][1] = -exp(mks_smooth * (startx1 - Xnative[1])) * mks_smooth
* (
M_PI / 2. -
M_PI * Xnative[2]
+ poly_norm * (2. * Xnative[2] - 1.)
* (1
+ (m::pow((-1. + 2 * Xnative[2]) / poly_xt, poly_alpha))
/ (1 + poly_alpha))
- 1. / 2. * (1. - hslope) * sin(2. * M_PI * Xnative[2]));
dxdX[2][2] = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2])
+ exp(mks_smooth * (startx1 - Xnative[1]))
* (-M_PI
+ 2. * poly_norm
* (1.
+ m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha)
/ (poly_alpha + 1.))
+ (2. * poly_alpha * poly_norm * (2. * Xnative[2] - 1.)
* m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha - 1.))
/ ((1. + poly_alpha) * poly_xt)
- (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2]));
dxdX[3][3] = 1.;
}

KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
{
Real dxdX_tmp[GR_DIM][GR_DIM];
dxdX(Xnative, dxdX_tmp);
invert(&dxdX_tmp[0][0],&dXdx[0][0]);
}
};

using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords>;
using SomeTransform = mpark::variant<NullTransform, ExponentialTransform, ModifyTransform, FunkyTransform>;
