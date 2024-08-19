
#pragma once

#include <complex>

#include "decs.hpp"


using namespace std::literals::complex_literals;
using namespace parthenon;


TaskStatus InitializeExplosion(MeshBlockData<Real> *rc, ParameterInput *pin)
{
auto pmb = rc->GetBlockPointer();

GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P = rc->Get("prims.B").data;

const auto& G = pmb->coords;

Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

const bool linear_ramp = pin->GetOrAddBoolean("explosion", "linear_ramp", false);
const Real u_out = pin->GetOrAddReal("explosion", "u_out", 3.e-5 / (gam-1));
const Real rho_out = pin->GetOrAddReal("explosion", "rho_out", 1.e-4);
const Real u_in = pin->GetOrAddReal("explosion", "u_in", 1.0 / (gam-1));
const Real rho_in = pin->GetOrAddReal("explosion", "rho_in", 1.e-2);

const Real r_in = pin->GetOrAddReal("explosion", "r_in", 0.8);
const Real r_out = pin->GetOrAddReal("explosion", "r_out", 1.0);
const Real xoff = pin->GetOrAddReal("explosion", "xoff", 0.0);
const Real yoff = pin->GetOrAddReal("explosion", "yoff", 0.0);

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
pmb->par_for("explosion_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);
const GReal rx = X[1] - xoff;
const GReal ry = X[2] - yoff;
const Real r = m::sqrt(rx*rx + ry*ry);

if (r < r_in) {
rho(k, j, i) = rho_in;
u(k, j, i) = u_in;
} else if (r >= r_in && r <= r_out) {
const Real ramp = (r_out - r) / (r_out - r_in);

if (linear_ramp) {
rho(k, j, i) = rho_out + ramp * (rho_in - rho_out);
u(k, j, i) = u_in + ramp * (u_in - u_out);
} else {
const Real lrho_out = log(rho_out);
const Real lrho_in = log(rho_in);
const Real lu_out = log(u_out);
const Real lu_in = log(u_in);
rho(k, j, i) = exp(lrho_out + ramp * (lrho_in - lrho_out));
u(k, j, i) = exp(lu_out + ramp * (lu_in - lu_out));
}
} else {
rho(k, j, i) = rho_out;
u(k, j, i) = u_out;
}
}
);

return TaskStatus::complete;
}
