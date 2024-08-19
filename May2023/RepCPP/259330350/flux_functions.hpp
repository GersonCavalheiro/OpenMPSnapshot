
#pragma once

#include "decs.hpp"

#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"
#include "types.hpp"


namespace Flux
{

template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& dir,
Real T[GR_DIM])
{
if (m_p.Q >= 0) {

Real q, dP;
const Real Theta = (gam - 1) * P(m_p.UU) / P(m_p.RHO);
const Real cs2   = gam * (gam - 1) * P(m_p.UU) / (P(m_p.RHO) + gam * P(m_p.UU));
EMHD::convert_prims_to_q_dP(P(m_p.Q), P(m_p.DP), P(m_p.RHO), Theta, cs2, emhd_params, q, dP);

EMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), q, dP, D, dir, T);
} else if (m_p.B1 >= 0) {
GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
} else {
GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
}

}

template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
const Real& gam, const int& dir,
Real T[GR_DIM])
{
if (m_p.B1 >= 0) {
GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
} else {
GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
}
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Global& P, const VarMap& m_p, const FourVectors D,
const Real& gam, const int& k, const int& j, const int& i, const int& dir,
Real T[GR_DIM])
{
if (m_p.B1 >= 0) {
GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
} else {
GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
}
}


template<typename Local>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& j, const int& i, const int& dir,
const Local& flux, const VarMap& m_u, const Loci loc=Loci::center)
{
Real gdet = G.gdet(loc, j, i);
flux(m_u.RHO) = P(m_p.RHO) * D.ucon[dir] * gdet;

Real T[GR_DIM];
calc_tensor(G, P, m_p, D, emhd_params, gam, dir, T);
flux(m_u.UU) = T[0] * gdet + flux(m_u.RHO);
flux(m_u.U1) = T[1] * gdet;
flux(m_u.U2) = T[2] * gdet;
flux(m_u.U3) = T[3] * gdet;

if (m_p.B1 >= 0) {
if (dir == 0) {
VLOOP flux(m_u.B1 + v) = P(m_p.B1 + v) * gdet;
} else {
VLOOP flux(m_u.B1 + v) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
}
if (m_p.PSI >= 0) {
if (dir == 0) {
flux(m_u.PSI) = P(m_p.PSI) * gdet;
} else {
flux(m_u.PSI) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI)) * gdet;
}
}
}

if (m_p.Q >= 0) {
flux(m_u.Q) = P(m_p.Q) * D.ucon[dir] * gdet;
flux(m_u.DP) = P(m_p.DP) * D.ucon[dir] * gdet;
}

if (m_p.KTOT >= 0) {
flux(m_u.KTOT) = flux(m_u.RHO) * P(m_p.KTOT);
if (m_p.K_CONSTANT >= 0)
flux(m_u.K_CONSTANT) = flux(m_u.RHO) * P(m_p.K_CONSTANT);
if (m_p.K_HOWES >= 0)
flux(m_u.K_HOWES) = flux(m_u.RHO) * P(m_p.K_HOWES);
if (m_p.K_KAWAZURA >= 0)
flux(m_u.K_KAWAZURA) = flux(m_u.RHO) * P(m_p.K_KAWAZURA);
if (m_p.K_WERNER >= 0)
flux(m_u.K_WERNER) = flux(m_u.RHO) * P(m_p.K_WERNER);
if (m_p.K_ROWAN >= 0)
flux(m_u.K_ROWAN) = flux(m_u.RHO) * P(m_p.K_ROWAN);
if (m_p.K_SHARMA >= 0)
flux(m_u.K_SHARMA) = flux(m_u.RHO) * P(m_p.K_SHARMA);
}

}

template<typename Global>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Global& P, const VarMap& m_p, const FourVectors D,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, 
const int& k, const int& j, const int& i, const int dir,
const Global& flux, const VarMap& m_u, const Loci loc=Loci::center)
{
const Real gdet = G.gdet(loc, j, i);
flux(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * D.ucon[dir] * gdet;

Real T[GR_DIM];
if (m_p.Q >= 0) {

Real q, dP;
const Real Theta = (gam - 1) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
const Real cs2   = gam * (gam - 1) * P(m_p.UU, k, j, i) / (P(m_p.RHO, k, j, i) + gam * P(m_p.UU, k, j, i));
EMHD::convert_prims_to_q_dP(P(m_p.Q, k, j, i), P(m_p.DP, k, j, i), P(m_p.RHO, k, j, i), Theta, cs2, emhd_params, q, dP);

EMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), q, dP, D, dir, T);
} else if (m_p.B1 >= 0) {
GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
} else {
GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
}
flux(m_u.UU, k, j, i) = T[0] * gdet + flux(m_u.RHO, k, j, i);
flux(m_u.U1, k, j, i) = T[1] * gdet;
flux(m_u.U2, k, j, i) = T[2] * gdet;
flux(m_u.U3, k, j, i) = T[3] * gdet;

if (m_p.B1 >= 0) {
if (dir == 0) {
VLOOP flux(m_u.B1 + v, k, j, i) = P(m_p.B1 + v, k, j, i) * gdet;
} else {
VLOOP flux(m_u.B1 + v, k, j, i) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
}
if (m_p.PSI >= 0) {
if (dir == 0) {
flux(m_u.PSI, k, j, i) = P(m_p.PSI, k, j, i) * gdet;
} else {
flux(m_u.PSI, k, j, i) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI, k, j, i)) * gdet;
}
}
}

if (m_p.Q >= 0) {
flux(m_u.Q, k, j, i)  = P(m_p.Q, k, j, i) * D.ucon[dir] * gdet;
flux(m_u.DP, k, j, i) = P(m_p.DP, k, j, i) * D.ucon[dir] * gdet;
}

if (m_p.KTOT >= 0) {
flux(m_u.KTOT, k, j, i)  = flux(m_u.RHO, k, j, i) * P(m_p.KTOT, k, j, i);
if (m_p.K_CONSTANT >= 0)
flux(m_u.K_CONSTANT, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_CONSTANT, k, j, i);
if (m_p.K_HOWES >= 0)
flux(m_u.K_HOWES, k, j, i)    = flux(m_u.RHO, k, j, i) * P(m_p.K_HOWES, k, j, i);
if (m_p.K_KAWAZURA >= 0)
flux(m_u.K_KAWAZURA, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_KAWAZURA, k, j, i);
if (m_p.K_WERNER >= 0)
flux(m_u.K_WERNER, k, j, i)   = flux(m_u.RHO, k, j, i) * P(m_p.K_WERNER, k, j, i);
if (m_p.K_ROWAN >= 0)
flux(m_u.K_ROWAN, k, j, i)    = flux(m_u.RHO, k, j, i) * P(m_p.K_ROWAN, k, j, i);
if (m_p.K_SHARMA >= 0)
flux(m_u.K_SHARMA, k, j, i)   = flux(m_u.RHO, k, j, i) * P(m_p.K_SHARMA, k, j, i);
}

}


template<typename Local>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Local& P, const VarMap& m_p,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& j, const int& i,
const Local& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, j, i, loc, Dtmp); 
prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, j, i, 0, U, m_u, loc);
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Global& P, const VarMap& m_p,
const EMHD::EMHD_parameters& emhd_params, const Real& gam, 
const int& k, const int& j, const int& i,
const Global& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, k, j, i, 0, U, m_u, loc);
}


template<typename Local>
KOKKOS_INLINE_FUNCTION void vchar(const GRCoordinates& G, const Local& P, const VarMap& m, const FourVectors& D,
const Real& gam, const EMHD::EMHD_parameters& emhd_params, 
const int& k, const int& j, const int& i, const Loci& loc, const int& dir,
Real& cmax, Real& cmin)
{
const Real ef  = P(m.RHO) + gam * P(m.UU);
const Real cs2 = gam * (gam - 1) * P(m.UU) / ef;
Real cms2;
if (m.Q > 0) {
const Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);
const Real ee  = bsq + ef;
const Real va2 = bsq / ee;

const Real cvis2  = (4./3.) / (P(m.RHO) + (gam * P(m.UU)) ) * P(m.RHO) * emhd_params.viscosity_alpha * cs2;
const Real ccond2 = (gam - 1.) * emhd_params.conduction_alpha * cs2;

const Real cscond   = 0.5*(cs2 + ccond2 + sqrt(cs2*cs2 + ccond2*ccond2) ) ;
const Real cs2_emhd = cscond + cvis2;

cms2 = cs2_emhd + va2 - cs2_emhd*va2;
} else if (m.B1 >= 0) {
const Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);
const Real ee  = bsq + ef;
const Real va2 = bsq / ee;

cms2 = cs2 + va2 - cs2 * va2;
} else {
cms2 = cs2;
}
clip(cms2, SMALL, 1.);

Real A, B, C;
{
Real Bcov[GR_DIM] = {1., 0., 0., 0.};
Real Acov[GR_DIM] = {0}; Acov[dir] = 1.;

Real Acon[GR_DIM], Bcon[GR_DIM];
G.raise(Acov, Acon, k, j, i, loc);
G.raise(Bcov, Bcon, k, j, i, loc);

const Real Asq  = dot(Acon, Acov);
const Real Bsq  = dot(Bcon, Bcov);
const Real Au   = dot(Acov, D.ucon);
const Real Bu   = dot(Bcov, D.ucon);
const Real AB   = dot(Acon, Bcov);
const Real Au2  = Au * Au;
const Real Bu2  = Bu * Bu;
const Real AuBu = Au * Bu;

A = Bu2 - (Bsq + Bu2) * cms2;
B = 2. * (AuBu - (AB + AuBu) * cms2);
C = Au2 - (Asq + Au2) * cms2;
}

Real discr = m::sqrt(m::max(B * B - 4. * A * C, 0.));

Real vp = -(-B + discr) / (2. * A);
Real vm = -(-B - discr) / (2. * A);

cmax = m::max(vp, vm);
cmin = m::min(vp, vm);
}

} 
