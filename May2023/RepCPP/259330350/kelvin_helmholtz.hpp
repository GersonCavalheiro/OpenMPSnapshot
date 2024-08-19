
#pragma once

#include "decs.hpp"

#include <parthenon/parthenon.hpp>


TaskStatus InitializeKelvinHelmholtz(MeshBlockData<Real> *rc, ParameterInput *pin)
{
auto pmb = rc->GetBlockPointer();
GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P = rc->Get("prims.B").data;

const Real tscale = pin->GetOrAddReal("kelvin_helmholtz", "tscale", 0.05);
const Real rho0 = pin->GetOrAddReal("kelvin_helmholtz", "rho0", 1.);
const Real Drho = pin->GetOrAddReal("kelvin_helmholtz", "Drho", 0.1);
const Real P0 = pin->GetOrAddReal("kelvin_helmholtz", "P0", 10.);
const Real uflow = pin->GetOrAddReal("kelvin_helmholtz", "uflow", 1.);
const Real a = pin->GetOrAddReal("kelvin_helmholtz", "a", 0.05);
const Real sigma = pin->GetOrAddReal("kelvin_helmholtz", "sigma", 0.2);
const Real A = pin->GetOrAddReal("kelvin_helmholtz", "A", 0.01);
const Real z1 = pin->GetOrAddReal("kelvin_helmholtz", "z1", 0.5);
const Real z2 = pin->GetOrAddReal("kelvin_helmholtz", "z2", 1.5);

const auto& G = pmb->coords;
const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
pmb->par_for("kh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
GReal X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);

GReal x = X[1];
GReal z = X[2];

rho(k, j, i) =
rho0 + Drho * 0.5 * (tanh((z - z1) / a) - tanh((z - z2) / a));
u(k, j, i) = P0 / (gam - 1.);
uvec(0, k, j, i) = uflow * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1.);
uvec(1, k, j, i) = A * sin(2. * M_PI * x) *
(exp(-(z - z1) * (z - z1) / (sigma * sigma)) +
exp(-(z - z2) * (z - z2) / (sigma * sigma)));
uvec(2, k, j, i) = 0;
}
);
pmb->par_for("kh_renorm", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
u(k, j, i) *= tscale * tscale;
VLOOP uvec(v, k, j, i) *= tscale;
}
);

return TaskStatus::complete;
}
