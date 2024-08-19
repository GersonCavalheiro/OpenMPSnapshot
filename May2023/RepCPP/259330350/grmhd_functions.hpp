
#pragma once

#include "decs.hpp"

#include "gr_coordinates.hpp"
#include "types.hpp"
#include "kharma_utils.hpp"


namespace GRHD
{

KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
const FourVectors& D, const int dir,
Real hd[GR_DIM])
{
const Real eta = pgas + rho + u;
DLOOP1 {
hd[mu] = eta * D.ucon[dir] * D.ucov[mu] +
pgas * (dir == mu);
}
}

}


namespace GRMHD
{


KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const GridVector uvec,
const int& k, const int& j, const int& i,
const Loci loc)
{

const Real qsq = G.gcov(loc, j, i, 1, 1) * uvec(V1, k, j, i) * uvec(V1, k, j, i) +
G.gcov(loc, j, i, 2, 2) * uvec(V2, k, j, i) * uvec(V2, k, j, i) +
G.gcov(loc, j, i, 3, 3) * uvec(V3, k, j, i) * uvec(V3, k, j, i) +
2. * (G.gcov(loc, j, i, 1, 2) * uvec(V1, k, j, i) * uvec(V2, k, j, i) +
G.gcov(loc, j, i, 1, 3) * uvec(V1, k, j, i) * uvec(V3, k, j, i) +
G.gcov(loc, j, i, 2, 3) * uvec(V2, k, j, i) * uvec(V3, k, j, i));

return m::sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const Real uv[NVEC],
const int& k, const int& j, const int& i,
const Loci loc)
{
const Real qsq = G.gcov(loc, j, i, 1, 1) * uv[V1] * uv[V1] +
G.gcov(loc, j, i, 2, 2) * uv[V2] * uv[V2] +
G.gcov(loc, j, i, 3, 3) * uv[V3] * uv[V3] +
2. * (G.gcov(loc, j, i, 1, 2) * uv[V1] * uv[V2] +
G.gcov(loc, j, i, 1, 3) * uv[V1] * uv[V3] +
G.gcov(loc, j, i, 2, 3) * uv[V2] * uv[V3]);

return m::sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m,
const int& k, const int& j, const int& i, const Loci& loc=Loci::center)
{
const Real qsq = G.gcov(loc, j, i, 1, 1) * P(m.U1, k, j, i) * P(m.U1, k, j, i) +
G.gcov(loc, j, i, 2, 2) * P(m.U2, k, j, i) * P(m.U2, k, j, i) +
G.gcov(loc, j, i, 3, 3) * P(m.U3, k, j, i) * P(m.U3, k, j, i) +
2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1, k, j, i) * P(m.U2, k, j, i) +
G.gcov(loc, j, i, 1, 3) * P(m.U1, k, j, i) * P(m.U3, k, j, i) +
G.gcov(loc, j, i, 2, 3) * P(m.U2, k, j, i) * P(m.U3, k, j, i));

return m::sqrt(1. + qsq);
}
template<typename Local>
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const Local& P, const VarMap& m,
const int& j, const int& i, const Loci& loc=Loci::center)
{
const Real qsq = G.gcov(loc, j, i, 1, 1) * P(m.U1) * P(m.U1) +
G.gcov(loc, j, i, 2, 2) * P(m.U2) * P(m.U2) +
G.gcov(loc, j, i, 3, 3) * P(m.U3) * P(m.U3) +
2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1) * P(m.U2) +
G.gcov(loc, j, i, 1, 3) * P(m.U1) * P(m.U3) +
G.gcov(loc, j, i, 2, 3) * P(m.U2) * P(m.U3));

return m::sqrt(1. + qsq);
}


KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
const FourVectors& D, const int dir,
Real mhd[GR_DIM])
{
const Real bsq = dot(D.bcon, D.bcov);
const Real eta = pgas + rho + u + bsq;
const Real ptot = pgas + 0.5 * bsq;

DLOOP1 {
mhd[mu] = eta * D.ucon[dir] * D.ucov[mu] +
ptot * (dir == mu) -
D.bcon[dir] * D.bcov[mu];
}
}


KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Real uvec[NVEC], const Real B_P[NVEC],
const int& k, const int& j, const int& i, const Loci loc,
FourVectors& D)
{
const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

D.ucon[0] = gamma / alpha;
VLOOP D.ucon[v+1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

G.lower(D.ucon, D.ucov, k, j, i, loc);

D.bcon[0] = 0;
VLOOP D.bcon[0] += B_P[v] * D.ucov[v+1];
VLOOP D.bcon[v+1] = (B_P[v] + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const GridVector uvec, const GridVector B_P,
const int& k, const int& j, const int& i, const Loci loc,
FourVectors& D)
{
const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

D.ucon[0] = gamma / alpha;
VLOOP D.ucon[v+1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

G.lower(D.ucon, D.ucov, k, j, i, loc);

D.bcon[0] = 0;
VLOOP D.bcon[0] += B_P(v, k, j, i) * D.ucov[v+1];
VLOOP D.bcon[v+1] = (B_P(v, k, j, i) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

G.lower(D.bcon, D.bcov, k, j, i, loc);
}
template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Local& P, const VarMap& m,
const int& j, const int& i, const Loci loc, FourVectors& D)
{
const Real gamma = lorentz_calc(G, P, m, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

D.ucon[0] = gamma / alpha;
VLOOP D.ucon[v+1] = P(m.U1 + v) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

G.lower(D.ucon, D.ucov, 0, j, i, loc);

if (m.B1 >= 0) {
D.bcon[0] = 0;
VLOOP D.bcon[0] += P(m.B1 + v) * D.ucov[v+1];
VLOOP D.bcon[v+1] = (P(m.B1 + v) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

G.lower(D.bcon, D.bcov, 0, j, i, loc);
} else {
DLOOP1 D.bcon[mu] = D.bcov[mu] = 0.;
}
}
template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Global& P, const VarMap& m,
const int& k, const int& j, const int& i, const Loci loc, FourVectors& D)
{
const Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

D.ucon[0] = gamma / alpha;
VLOOP D.ucon[v+1] = P(m.U1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

G.lower(D.ucon, D.ucov, k, j, i, loc);

if (m.B1 >= 0) {
D.bcon[0] = 0;
VLOOP D.bcon[0]  += P(m.B1 + v, k, j, i) * D.ucov[v+1];
VLOOP D.bcon[v+1] = (P(m.B1 + v, k, j, i) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

G.lower(D.bcon, D.bcov, k, j, i, loc);
} else {
DLOOP1 D.bcon[mu] = D.bcov[mu] = 0.;
}
}

KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates &G, const GridVector uvec,
const int& k, const int& j, const int& i, const Loci loc,
Real ucon[GR_DIM])
{
const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

ucon[0] = gamma / alpha;
VLOOP ucon[v+1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates &G, const Real uvec[NVEC],
const int& k, const int& j, const int& i, const Loci loc,
Real ucon[GR_DIM])
{
const Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

ucon[0] = gamma / alpha;
VLOOP ucon[v+1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}
template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates& G, const Local& P, const VarMap& m,
const int& j, const int& i, const Loci loc,
Real ucon[GR_DIM])
{
const Real gamma = lorentz_calc(G, P, m, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

ucon[0] = gamma / alpha;
VLOOP ucon[v+1] = P(m.U1 + v) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}
template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates& G, const Global& P, const VarMap& m,
const int& k, const int& j, const int& i, const Loci loc,
Real ucon[GR_DIM])
{
const Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
const Real alpha = 1. / m::sqrt(-G.gcon(loc, j, i, 0, 0));

ucon[0] = gamma / alpha;
VLOOP ucon[v+1] = P(m.U1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}


template<typename Local>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Local& P, const VarMap& m_p,
const Real& gam, const int& j, const int& i,
const Local& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
Real gdet = G.gdet(loc, j, i);
FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, j, i, loc, Dtmp); 
U(m_u.RHO) = P(m_p.RHO) * Dtmp.ucon[0] * gdet;

if (m_p.B1 >= 0) {
Real mhd[GR_DIM];
GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), Dtmp, 0, mhd);
U(m_u.UU)  = mhd[0] * gdet + U(m_u.RHO);
U(m_u.U1) =  mhd[1] * gdet;
U(m_u.U2) =  mhd[2] * gdet;
U(m_u.U3) =  mhd[3] * gdet;
} else {
Real hd[GR_DIM];
GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), Dtmp, 0, hd);
U(m_u.UU) = hd[0] * gdet + U(m_u.RHO);
U(m_u.U1) = hd[1] * gdet;
U(m_u.U2) = hd[2] * gdet;
U(m_u.U3) = hd[3] * gdet;
}
}
template<typename Global>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Global& P, const VarMap& m_p,
const Real& gam, const int& k, const int& j, const int& i,
const Global& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
Real gdet = G.gdet(loc, j, i);
FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp); 
U(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * Dtmp.ucon[0] * gdet;

if (m_p.B1 >= 0) {
Real mhd[GR_DIM];
GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), Dtmp, 0, mhd);
U(m_u.UU, k, j, i)  = mhd[0] * gdet + U(m_u.RHO, k, j, i);
U(m_u.U1, k, j, i) =  mhd[1] * gdet;
U(m_u.U2, k, j, i) =  mhd[2] * gdet;
U(m_u.U3, k, j, i) =  mhd[3] * gdet;
} else {
Real hd[GR_DIM];
GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), Dtmp, 0, hd);
U(m_u.UU, k, j, i) = hd[0] * gdet + U(m_u.RHO, k, j, i);
U(m_u.U1, k, j, i) = hd[1] * gdet;
U(m_u.U2, k, j, i) = hd[2] * gdet;
U(m_u.U3, k, j, i) = hd[3] * gdet;
}
}


KOKKOS_INLINE_FUNCTION void p_to_u_mhd(const GRCoordinates& G, const Real& rho, const Real& u, const Real uvec[NVEC],
const Real B_P[NVEC], const Real& gam, const int& k, const int& j, const int& i,
Real& rho_ut, Real T[GR_DIM], const Loci loc=Loci::center)
{
Real gdet = G.gdet(loc, j, i);

FourVectors Dtmp;
calc_4vecs(G, uvec, B_P, k, j, i, loc, Dtmp);

rho_ut = rho * Dtmp.ucon[0] * gdet;

Real mhd[GR_DIM];
calc_tensor(rho, u, (gam - 1) * u, Dtmp, 0, mhd);

T[0]  = mhd[0] * gdet + rho_ut;
VLOOP T[1 + v] = mhd[1 + v] * gdet;
}

}
