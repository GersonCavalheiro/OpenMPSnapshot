#pragma once

#include "decs.hpp"

using namespace parthenon;


TaskStatus InitializeOrszagTang(MeshBlockData<Real> *rc, ParameterInput *pin)
{
Flag(rc, "Initializing Orszag-Tang problem");
auto pmb = rc->GetBlockPointer();
GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P = rc->Get("prims.B").data;

const auto& G = pmb->coords;

const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
const Real tscale = pin->GetOrAddReal("orszag_tang", "tscale", 0.05);
const Real phase = pin->GetOrAddReal("orszag_tang", "phase", M_PI);

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
Real X[GR_DIM];
G.coord(k, j, i, Loci::center, X);
rho(k, j, i) = 25./9.;
u(k, j, i) = 5./(3.*(gam - 1.));
uvec(0, k, j, i) = -sin(X[2] + phase);
uvec(1, k, j, i) = sin(X[1] + phase);
uvec(2, k, j, i) = 0.;
B_P(0, k, j, i) = -sin(X[2] + phase);
B_P(1, k, j, i) = sin(2.*(X[1] + phase));
B_P(2, k, j, i) = 0.;
}
);
pmb->par_for("ot_renorm", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
u(k, j, i) *= tscale * tscale;
VLOOP uvec(v, k, j, i) *= tscale;
VLOOP B_P(v, k, j, i) *= tscale;
}
);

return TaskStatus::complete;
}
