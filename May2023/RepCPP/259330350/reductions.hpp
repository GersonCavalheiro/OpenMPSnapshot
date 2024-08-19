
#pragma once

#include "debug.hpp"

#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

namespace Reductions {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);



template<typename T>
Real AccretionRate(MeshData<Real> *md, const int& i);
template<typename T>
Real DomainSum(MeshData<Real> *md, const Real& radius);


#define MAKE_SUM2D_FN(name, fn) template<> inline Real AccretionRate<name>(MeshData<Real> *md, const int& i) { \
Flag("Performing accretion reduction"); \
auto pmesh = md->GetMeshPointer(); \
\
Real result = 0.; \
for (auto &pmb : pmesh->block_list) { \
auto& rc = pmb->meshblock_data.Get(); \
if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) { \
const auto& pars = pmb->packages.Get("GRMHD")->AllParams(); \
const MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag"); \
PackIndexMap prims_map, cons_map; \
const auto& P = rc->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map); \
const auto& U = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map); \
const VarMap m_u(cons_map, true), m_p(prims_map, false); \
\
const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
const auto& G = pmb->coords; \
\
Real block_result; \
Kokkos::Sum<Real> sum_reducer(block_result); \
pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+i, ib.s+i, \
KOKKOS_LAMBDA_3D_REDUCE { \
FourVectors Dtmp; \
Real T[GR_DIM][GR_DIM]; \
GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp); \
DLOOP1 Flux::calc_tensor(G, P, m_p, Dtmp, gam, k, j, i, mu, T[mu]); \
GReal gdA = G.dx3v(k) * G.dx2v(j) * G.gdet(Loci::center, j, i); \
GReal dA = G.dx3v(k) * G.dx2v(j); \
fn \
} \
, sum_reducer); \
result += block_result; \
} \
} \
\
Flag("Reduced"); \
\
return result; \
}


enum class Mdot : int;
MAKE_SUM2D_FN(Mdot,
local_result += -P(m_p.RHO, k, j, i) * Dtmp.ucon[1] * gdA;
)
enum class Edot : int;
MAKE_SUM2D_FN(Edot,
local_result += -T[X1DIR][X0DIR] * gdA;
)
enum class Ldot : int;
MAKE_SUM2D_FN(Ldot,
local_result += T[X1DIR][X3DIR] * gdA;
)
enum class Phi : int;
MAKE_SUM2D_FN(Phi,
if (m_u.B1 >= 0) {
local_result += 0.5 * m::abs(U(m_u.B1, k, j, i)) * dA; 
}
)

enum class Mdot_Flux : int;
MAKE_SUM2D_FN(Mdot_Flux, local_result += -U.flux(X1DIR, m_u.RHO, k, j, i) * dA;)
enum class Edot_Flux : int;
MAKE_SUM2D_FN(Edot_Flux, local_result += (U.flux(X1DIR, m_u.UU, k, j, i) - U.flux(X1DIR, m_u.RHO, k, j, i)) * dA;)
enum class Ldot_Flux : int;
MAKE_SUM2D_FN(Ldot_Flux, local_result += U.flux(X1DIR, m_u.U3, k, j, i) * dA;)

inline Real MdotBound(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 0);}
inline Real MdotEH(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 5);}
inline Real EdotBound(MeshData<Real> *md) {return AccretionRate<Edot>(md, 0);}
inline Real EdotEH(MeshData<Real> *md) {return AccretionRate<Edot>(md, 5);}
inline Real LdotBound(MeshData<Real> *md) {return AccretionRate<Ldot>(md, 0);}
inline Real LdotEH(MeshData<Real> *md) {return AccretionRate<Ldot>(md, 5);}
inline Real PhiBound(MeshData<Real> *md) {return AccretionRate<Phi>(md, 0);}
inline Real PhiEH(MeshData<Real> *md) {return AccretionRate<Phi>(md, 5);}

inline Real MdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 0);}
inline Real MdotEHFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 5);}
inline Real EdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 0);}
inline Real EdotEHFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 5);}
inline Real LdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 0);}
inline Real LdotEHFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 5);}


#define MAKE_SUM3D_FN(name, fn) template<> inline Real DomainSum<name>(MeshData<Real> *md, const Real& radius) { \
Flag("Performing domain reduction"); \
auto pmesh = md->GetMeshPointer(); \
\
Real result = 0.; \
for (auto &pmb : pmesh->block_list) { \
auto& rc = pmb->meshblock_data.Get(); \
if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) { \
const auto& pars = pmb->packages.Get("GRMHD")->AllParams(); \
const MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag"); \
PackIndexMap prims_map, cons_map; \
const auto& P = rc->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map); \
const auto& U = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map); \
const VarMap m_u(cons_map, true), m_p(prims_map, false); \
\
const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
const auto& G = pmb->coords; \
\
Real block_result; \
Kokkos::Sum<Real> sum_reducer(block_result); \
pmb->par_reduce("domain_sum", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, \
KOKKOS_LAMBDA_3D_REDUCE { \
FourVectors Dtmp; \
Real T[GR_DIM][GR_DIM]; \
GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp); \
DLOOP1 Flux::calc_tensor(G, P, m_p, Dtmp, gam, k, j, i, mu, T[mu]); \
GReal gdV = G.dx3v(k) * G.dx2v(j) * G.dx1v(i) * G.gdet(Loci::center, j, i); \
GReal dV = G.dx3v(k) * G.dx2v(j) * G.dx1v(i); \
fn \
} \
, sum_reducer); \
result += block_result; \
} \
} \
\
Flag("Reduced"); \
\
return result; \
}
enum class Mtot : int;
MAKE_SUM3D_FN(Mtot,
GReal X[GR_DIM];
G.coord_embed(k, j, i, Loci::face1, X);
if (X[1] < radius) {
local_result += U(m_u.RHO, k, j, i) * dV;
}
)
enum class Ltot : int;
MAKE_SUM3D_FN(Ltot,
GReal X[GR_DIM];
G.coord_embed(k, j, i, Loci::face1, X);
if (X[1] < radius) {
local_result += U(m_u.U3, k, j, i) * dV;
}
)
enum class Etot : int;
MAKE_SUM3D_FN(Etot,
GReal X[GR_DIM];
G.coord_embed(k, j, i, Loci::face1, X);
if (X[1] < radius) {
local_result += U(m_u.UU, k, j, i) * dV;
}
)

enum class EHTLum : int;
MAKE_SUM3D_FN(EHTLum,
GReal X[GR_DIM];
G.coord_embed(k, j, i, Loci::face1, X);
if (X[1] > radius) {
Real rho = P(m_p.RHO, k, j, i);
Real Pg = (gam - 1.) * P(m_p.UU, k, j, i);
Real Bmag = m::sqrt(dot(Dtmp.bcon, Dtmp.bcov));
Real j_eht = m::pow(rho, 3.) * m::pow(Pg, -2.) * exp(-0.2 * m::pow(rho * rho / (Bmag * Pg * Pg), 1./3.));
local_result += j_eht * gdV;
}
)

enum class JetLum : int;
MAKE_SUM3D_FN(JetLum,
GReal X_f[GR_DIM]; GReal X_b[GR_DIM];
G.coord_embed(k, j, i, Loci::face1, X_b);
G.coord_embed(k, j, i+1, Loci::face1, X_f);
if (X_f[1] > radius && X_b[1] < radius) {
if ((dot(Dtmp.bcon, Dtmp.bcov) / P(m_p.RHO, k, j, i)) > 1.) {
local_result += -T[X1DIR][X0DIR] * G.dx3v(k) * G.dx2v(j) * G.gdet(Loci::center, j, i);;
}
}
)

inline Real TotalM(MeshData<Real> *md) {return DomainSum<Mtot>(md, 50.);}
inline Real TotalE(MeshData<Real> *md) {return DomainSum<Etot>(md, 50.);}
inline Real TotalL(MeshData<Real> *md) {return DomainSum<Ltot>(md, 50.);}

inline Real TotalEHTLum(MeshData<Real> *md) {return DomainSum<EHTLum>(md, 50.);}
inline Real JetLum_50(MeshData<Real> *md) {return DomainSum<JetLum>(md, 50.);} 



inline int NPFlags(MeshData<Real> *md) {return CountPFlags(md, IndexDomain::interior, 0);}
inline int NFFlags(MeshData<Real> *md) {return CountFFlags(md, IndexDomain::interior, 0);}

} 
