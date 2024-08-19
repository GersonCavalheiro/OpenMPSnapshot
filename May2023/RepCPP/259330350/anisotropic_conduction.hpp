
#pragma once

#include <complex>

#include "decs.hpp"

using namespace parthenon;


TaskStatus InitializeAnisotropicConduction(MeshBlockData<Real> *rc, ParameterInput *pin)
{
Flag(rc, "Initializing EMHD Modes problem");
auto pmb = rc->GetBlockPointer();
GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P = rc->Get("prims.B").data;
GridVector q = rc->Get("prims.q").data;
GridVector dP = rc->Get("prims.dP").data;

const auto& G = pmb->coords;

const Real A = pin->GetOrAddReal("anisotropic_conduction", "A", 0.2);
const Real Rsq = pin->GetOrAddReal("anisotropic_conduction", "Rsq", 0.005);
const Real B0 = pin->GetOrAddReal("anisotropic_conduction", "B0", 1e-4);
const Real k0 = pin->GetOrAddReal("anisotropic_conduction", "k", 4.);

const Real R = m::sqrt(Rsq);

IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
pmb->par_for("anisotropic_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);
GReal r = m::sqrt(m::pow((X[1] - 0.5), 2) + m::pow((X[2] - 0.5), 2));

rho(k, j, i) = 1 - (A * exp(-m::pow(r, 2) / m::pow(R, 2)));
u(k, j, i) = 1.;
uvec(0, k, j, i) = 0.;
uvec(1, k, j, i) = 0.;
uvec(2, k, j, i) = 0.;
B_P(0, k, j, i) = B0;
B_P(1, k, j, i) = B0 * sin(2*M_PI*k0*X[1]);
B_P(2, k, j, i) = 0;
q(k, j, i) = 0.;
dP(k, j, i) = 0.;
}
);

return TaskStatus::complete;
}
