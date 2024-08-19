
#pragma once

#include "decs.hpp"

#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "prob_common.hpp"
#include "types.hpp"
#include "emhd.hpp"

#include <parthenon/parthenon.hpp>


TaskStatus InitializeBondiViscous(MeshBlockData<Real> *rc, ParameterInput *pin);


TaskStatus SetBondiViscous(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);


KOKKOS_INLINE_FUNCTION Real get_Tfunc_viscous(const Real T, const GReal r, const Real C4, const Real C3, const Real n)
{
return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C4 / pow(r,2) / pow(T, n), 2.)) - C3;
}
KOKKOS_INLINE_FUNCTION Real get_T_viscous(const GReal r, const Real C4, const Real C3, const Real n)
{
Real rtol = 1.e-12;
Real ftol = 1.e-14;
Real Tmin = 0.6 * (sqrt(C3) - 1.) / (n + 1);
Real Tmax = pow(C4 * sqrt(2. / pow(r,3)), 1. / n);

Real f0, f1, fh;
Real T0, T1, Th;
T0 = Tmin;
f0 = get_Tfunc_viscous(T0, r, C4, C3, n);
T1 = Tmax;
f1 = get_Tfunc_viscous(T1, r, C4, C3, n);
if (f0 * f1 > 0) return -1;

Th = (f1 * T0 - f0 * T1) / (f1 - f0);
fh = get_Tfunc_viscous(Th, r, C4, C3, n);
Real epsT = rtol * (Tmin + Tmax);
while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol)
{
if (fh * f0 < 0.) {
T0 = Th;
f0 = fh;
} else {
T1 = Th;
f1 = fh;
}

Th = (f1 * T0 - f0 * T1) / (f1 - f0);
fh = get_Tfunc_viscous(Th, r, C4, C3, n);
}

return Th;
}


KOKKOS_INLINE_FUNCTION void get_prim_bondi_viscous(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
const Real mdot, const Real rs, const int& k, const int& j, const int& i)
{
Real n  = 1. / (gam - 1.);
Real uc = sqrt(1. / (2. * rs));
Real Vc = sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
Real C4 = uc * pow(rs, 2) * pow(Tc, n);
Real C3 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. / rs + pow(uc, 2));
Real K  = pow(4 * M_PI * C4 / mdot, 1/n);

GReal Xnative[GR_DIM], Xembed[GR_DIM];
G.coord(k, j, i, Loci::center, Xnative);
G.coord_embed(k, j, i, Loci::center, Xembed);
GReal r = Xembed[1];

Real T   = get_T_viscous(r, C4, C3, n);
Real ur  = -C4 / (pow(T, n) * pow(r, 2));
Real rho = pow(K, -n) * pow(T, n);
Real u   = rho * T / (gam - 1.);

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

P(m_p.RHO, k, j, i) = rho;
P(m_p.UU, k, j, i)  = u;
P(m_p.U1, k, j, i)  = u_prim[0];
P(m_p.U2, k, j, i)  = u_prim[1];
P(m_p.U3, k, j, i)  = u_prim[2];

P(m_p.B1, k, j, i) = 1. / pow(r, 3.);

}
