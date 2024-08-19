
#pragma once

#include "decs.hpp"

#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "prob_common.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>


TaskStatus InitializeBondi(MeshBlockData<Real> *rc, ParameterInput *pin);


TaskStatus SetBondi(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::interior, bool coarse=false);


KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
return m::pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + m::pow(C1 / m::pow(r,2) / m::pow(T, n), 2.)) - C2;
}
KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n)
{
Real rtol = 1.e-12;
Real ftol = 1.e-14;
Real Tmin = 0.6 * (m::sqrt(C2) - 1.) / (n + 1);
Real Tmax = m::pow(C1 * m::sqrt(2. / m::pow(r,3)), 1. / n);

Real f0, f1, fh;
Real T0, T1, Th;
T0 = Tmin;
f0 = get_Tfunc(T0, r, C1, C2, n);
T1 = Tmax;
f1 = get_Tfunc(T1, r, C1, C2, n);
if (f0 * f1 > 0) return -1;

Th = (f1 * T0 - f0 * T1) / (f1 - f0);
fh = get_Tfunc(Th, r, C1, C2, n);
Real epsT = rtol * (Tmin + Tmax);
while (m::abs(Th - T0) > epsT && m::abs(Th - T1) > epsT && m::abs(fh) > ftol)
{
if (fh * f0 < 0.) {
T0 = Th;
f0 = fh;
} else {
T1 = Th;
f1 = fh;
}

Th = (f1 * T0 - f0 * T1) / (f1 - f0);
fh = get_Tfunc(Th, r, C1, C2, n);
}

return Th;
}


KOKKOS_INLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
const Real mdot, const Real rs, const int& k, const int& j, const int& i)
{
Real n = 1. / (gam - 1.);
Real uc = m::sqrt(mdot / (2. * rs));
Real Vc = -m::sqrt(m::pow(uc, 2) / (1. - 3. * m::pow(uc, 2)));
Real Tc = -n * m::pow(Vc, 2) / ((n + 1.) * (n * m::pow(Vc, 2) - 1.));
Real C1 = uc * m::pow(rs, 2) * m::pow(Tc, n);
Real C2 = m::pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + m::pow(C1, 2) / (m::pow(rs, 4) * m::pow(Tc, 2 * n)));

GReal Xnative[GR_DIM], Xembed[GR_DIM];
G.coord(k, j, i, Loci::center, Xnative);
G.coord_embed(k, j, i, Loci::center, Xembed);
GReal r = Xembed[1];
if (ks.a > 0.1 && r < 2) return;

Real T = get_T(r, C1, C2, n);
Real ur = -C1 / (m::pow(T, n) * m::pow(r, 2));
Real rho = m::pow(T, n);
Real u = rho * T * n;

Real ucon_bl[GR_DIM] = {0, ur, 0, 0};
Real gcov_bl[GR_DIM][GR_DIM];
bl.gcov_embed(Xembed, gcov_bl);
set_ut(gcov_bl, ucon_bl);

Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
G.gcon(Loci::center, j, i, gcon);
fourvel_to_prim(gcon, ucon_mks, u_prim);

if (!isnan(rho)) P(m_p.RHO, k, j, i) = rho;
if (!isnan(u)) P(m_p.UU, k, j, i) = u;
if (!isnan(u_prim[0])) P(m_p.U1, k, j, i) = u_prim[0];
if (!isnan(u_prim[1])) P(m_p.U2, k, j, i) = u_prim[1];
if (!isnan(u_prim[2])) P(m_p.U3, k, j, i) = u_prim[2];
}
