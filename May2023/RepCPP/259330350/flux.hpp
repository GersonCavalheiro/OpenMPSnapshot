
#pragma once

#include "decs.hpp"

#include <parthenon/parthenon.hpp>

#include "debug.hpp"
#include "floors.hpp"
#include "flux_functions.hpp"
#include "pack.hpp"
#include "reconstruction.hpp"
#include "types.hpp"

#include "emhd.hpp"
#include "grmhd_functions.hpp"
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "electrons.hpp"

namespace Flux {

TaskStatus ApplyFluxes(MeshData<Real> *md, MeshData<Real> *mdudt);


TaskStatus PtoU(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::interior);
inline TaskStatus PtoUTask(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire) { return PtoU(rc, domain); }


KOKKOS_INLINE_FUNCTION Real llf(const Real& fluxL, const Real& fluxR, const Real& cmax, 
const Real& cmin, const Real& Ul, const Real& Ur)
{
Real ctop = m::max(cmax, cmin);
return 0.5 * (fluxL + fluxR - ctop * (Ur - Ul));
}
KOKKOS_INLINE_FUNCTION Real hlle(const Real& fluxL, const Real& fluxR, const Real& cmax,
const Real& cmin, const Real& Ul, const Real& Ur)
{
return (cmax*fluxL + cmin*fluxR - cmax*cmin*(Ur - Ul)) / (cmax + cmin);
}


template <ReconstructionType Recon, int dir>
inline TaskStatus GetFlux(MeshData<Real> *md)
{
Flag(md, "Recon and flux");
auto pmesh = md->GetMeshPointer();
auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
const int ndim = pmesh->ndim;
if (ndim < 3 && dir == X3DIR) return TaskStatus::complete;
if (ndim < 2 && dir == X2DIR) return TaskStatus::complete;

const auto& pars = pmb0->packages.Get("GRMHD")->AllParams();
const auto& globals = pmb0->packages.Get("Globals")->AllParams();
const auto& floor_pars = pmb0->packages.Get("Floors")->AllParams();
const bool use_hlle = pars.Get<bool>("use_hlle");
const bool reconstruction_floors = (Recon == ReconstructionType::weno5)
&& !floor_pars.Get<bool>("disable_floors");
const Floors::Prescription floors(floor_pars);
const auto& pkgs = pmb0->packages.AllPackages();
const bool use_b_flux_ct = pkgs.count("B_FluxCT");
const bool use_b_cd = pkgs.count("B_CD");
const bool use_electrons = pkgs.count("Electrons");
const bool use_emhd = pkgs.count("EMHD");
const MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag");

const Real gam = pars.Get<Real>("gamma");
const double ctop_max = (use_b_cd) ? globals.Get<Real>("ctop_max_last") : 0.0;

EMHD::EMHD_parameters emhd_params_tmp;
if (use_emhd) {
const auto& emhd_pars = pmb0->packages.Get("EMHD")->AllParams();
emhd_params_tmp = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
}
const EMHD::EMHD_parameters& emhd_params = emhd_params_tmp;

const Loci loc = loc_of(dir);

PackIndexMap prims_map, cons_map;
const auto& ctop = md->PackVariables(std::vector<std::string>{"ctop"});
const auto& P_all = md->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map);
const auto& U_all = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
const VarMap m_u(cons_map, true), m_p(prims_map, false);
Flag(md, "Packed variables");

const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
const IndexRange block = IndexRange{0, ctop.GetDim(5) - 1};
const int nvar = U_all.GetDim(4);
int halo = 1;
const IndexRange il = IndexRange{ib.s - halo, ib.e + halo};
const IndexRange jl = (ndim > 1) ? IndexRange{jb.s - halo, jb.e + halo} : jb;
const IndexRange kl = (ndim > 2) ? IndexRange{kb.s - halo, kb.e + halo} : kb;

const int scratch_level = 1; 
const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
const size_t speed_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(1, n1);
const size_t total_scratch_bytes = (6 + 1*(Recon != ReconstructionType::weno5) +
4*(Recon == ReconstructionType::linear_vl)) * var_size_in_bytes
+ 2 * speed_size_in_bytes;

Flag(md, "Flux kernel");
parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux", pmb0->exec_space,
total_scratch_bytes, scratch_level, block.s, block.e, kl.s, kl.e, jl.s, jl.e,
KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
const auto& G = U_all.GetCoords(b);
ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad2D<Real> Ul_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad2D<Real> Ur_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad2D<Real> Fl_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad2D<Real> Fr_s(member.team_scratch(scratch_level), nvar, n1);
ScratchPad1D<Real> cmax(member.team_scratch(scratch_level), n1);
ScratchPad1D<Real> cmin(member.team_scratch(scratch_level), n1);

KReconstruction::reconstruct<Recon, dir>(member, G, P_all(b), k, j, il.s, il.e, Pl_s, Pr_s);

member.team_barrier();

parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
auto Pl = Kokkos::subview(Pl_s, Kokkos::ALL(), i);
auto Pr = Kokkos::subview(Pr_s, Kokkos::ALL(), i);
if (reconstruction_floors) {
Floors::apply_geo_floors(G, Pl, m_p, gam, j, i, floors, loc);
Floors::apply_geo_floors(G, Pr, m_p, gam, j, i, floors, loc);
}
#if !FUSE_FLUX_KERNELS
}
);
member.team_barrier();

parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
auto Pl = Kokkos::subview(Pl_s, Kokkos::ALL(), i);
#endif
auto Ul = Kokkos::subview(Ul_s, Kokkos::ALL(), i);
auto Fl = Kokkos::subview(Fl_s, Kokkos::ALL(), i);
FourVectors Dtmp;

GRMHD::calc_4vecs(G, Pl, m_p, j, i, loc, Dtmp);
Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, 0, Ul, m_u, loc);
Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, dir, Fl, m_u, loc);

Real cmaxL, cminL;
Flux::vchar(G, Pl, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxL, cminL);

#if !FUSE_FLUX_KERNELS
cmax(i) = m::max(0., cmaxL);
cmin(i) = m::max(0., -cminL);
}
);
member.team_barrier();

parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
FourVectors Dtmp;
auto Pr = Kokkos::subview(Pr_s, Kokkos::ALL(), i);
#endif
auto Ur = Kokkos::subview(Ur_s, Kokkos::ALL(), i);
auto Fr = Kokkos::subview(Fr_s, Kokkos::ALL(), i);
GRMHD::calc_4vecs(G, Pr, m_p, j, i, loc, Dtmp);
Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, 0, Ur, m_u, loc);
Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, dir, Fr, m_u, loc);

Real cmaxR, cminR;
Flux::vchar(G, Pr, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxR, cminR);

#if FUSE_FLUX_KERNELS
cmax(i) = m::abs(m::max(cmaxL,  cmaxR));
cmin(i) = m::abs(m::max(-cminL, -cminR));

if (use_hlle) {
for (int p=0; p < nvar; ++p)
U_all(b).flux(dir, p, k, j, i) = hlle(Fl(p), Fr(p), cmax(i), cmin(i), Ul(p), Ur(p));
} else {
for (int p=0; p < nvar; ++p)
U_all(b).flux(dir, p, k, j, i) = llf(Fl(p), Fr(p), cmax(i), cmin(i), Ul(p), Ur(p));
}
if (use_b_cd) {
U_all(b).flux(dir, m_u.PSI, k, j, i) = llf(Fl(m_u.PSI), Fr(m_u.PSI), ctop_max, ctop_max, Ul(m_u.PSI), Ur(m_u.PSI));
U_all(b).flux(dir, m_u.B1+dir-1, k, j, i) = llf(Fl(m_u.B1+dir-1), Fr(m_u.B1+dir-1), ctop_max, ctop_max, Ul(m_u.B1+dir-1), Ur(m_u.B1+dir-1));
}
#else
cmax(i) = m::abs(m::max(cmax(i),  cmaxR));
cmin(i) = m::abs(m::max(cmin(i), -cminR));
#endif
ctop(b, dir-1, k, j, i) = m::max(cmax(i), cmin(i));
}
);
member.team_barrier();

#if !FUSE_FLUX_KERNELS
for (int p=0; p < nvar; ++p) {
if (use_b_cd && (p == m_u.PSI || p == m_u.B1+dir-1)) {
parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
U_all(b).flux(dir, p, k, j, i) = llf(Fl_s(p,i), Fr_s(p,i), ctop_max, ctop_max, Ul_s(p,i), Ur_s(p,i));
}
);
} else if (use_hlle) {
parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
U_all(b).flux(dir, p, k, j, i) = hlle(Fl_s(p,i), Fr_s(p,i), cmax(i), cmin(i), Ul_s(p,i), Ur_s(p,i));
}
);
} else {
parthenon::par_for_inner(member, il.s, il.e,
[&](const int& i) {
U_all(b).flux(dir, p, k, j, i) = llf(Fl_s(p,i), Fr_s(p,i), cmax(i), cmin(i), Ul_s(p,i), Ur_s(p,i));
}
);
}
}
#endif
}
);

Flag(md, "Finished recon and flux");
return TaskStatus::complete;
}
}
