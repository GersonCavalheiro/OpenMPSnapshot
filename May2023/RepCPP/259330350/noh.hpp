
#pragma once

#include "decs.hpp"

using namespace parthenon;


TaskStatus InitializeNoh(MeshBlockData<Real> *rc, ParameterInput *pin)
{
Flag(rc, "Initializing 1D (Noh) Shock test");
auto pmb = rc->GetBlockPointer();
GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridScalar ktot = rc->Get("prims.Ktot").data;
GridScalar kel_constant = rc->Get("prims.Kel_Constant").data;

const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");
const Real fel_constant = pmb->packages.Get("Electrons")->Param<Real>("fel_constant");

const Real mach = pin->GetOrAddReal("noh", "mach", 49);
const Real rhoL = pin->GetOrAddReal("noh", "rhoL", 1.0);
const Real rhoR = pin->GetOrAddReal("noh", "rhoR", 1.0);
const Real PL = pin->GetOrAddReal("noh", "PL", 0.1);
const Real PR = pin->GetOrAddReal("noh", "PR", 0.1);
bool set_tlim = pin->GetOrAddBoolean("noh", "set_tlim", false);

const auto& G = pmb->coords;

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
const Real center = (x1min + x1max) / 2.;

Real cs2 = (gam * (gam - 1) * PL) / rhoL;
Real v1 = mach * m::sqrt(cs2);

if (set_tlim) {
pin->SetReal("parthenon/time", "tlim", 0.6*(x1max - x1min)/v1);
}

double gamma = 1. / m::sqrt(1. - v1 * v1); 


pmb->par_for("noh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);

const bool lhs = X[1] < center;
rho(k, j, i) = (lhs) ? rhoL : rhoR;
u(k, j, i) = ((lhs) ? PL : PR)/(gam - 1.);
uvec(0, k, j, i) = ((lhs) ? v1 : -v1) * gamma;
uvec(1, k, j, i) = 0.0;
uvec(2, k, j, i) = 0.0;
}
);

Flag(rc, "Initialized 1D (Noh) Shock test");
return TaskStatus::complete;
}
