
#pragma once

#include <complex>

#include "decs.hpp"


using namespace std::literals::complex_literals;
using namespace parthenon;


TaskStatus InitializeMHDModes(MeshBlockData<Real> *rc, ParameterInput *pin)
{
Flag(rc, "Initializing MHD Modes problem");
auto pmb = rc->GetBlockPointer();
GridScalar rho = rc->Get("prims.rho").data;
GridScalar u = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P = rc->Get("prims.B").data;

const auto& G = pmb->coords;

const int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
const int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);
const bool one_period = pin->GetOrAddBoolean("mhdmodes", "one_period", true);

Real rho0 = 1.;
Real u0 = 1.;
Real u10 = 0.;
Real u20 = 0.;
Real u30 = 0.;

Real k1 = 2. * M_PI;
Real k2 = 2. * M_PI;
Real k3 = 2. * M_PI;
if (dir == 1)
k1 = 0;
if (dir == 2)
k2 = 0;
if (dir == 3)
k3 = 0;

Real amp = 1.e-4;

Real B10 = 0.;
Real B20 = 0.;
Real B30 = 0.;

std::complex<Real> omega;
Real drho = 0, du = 0;
Real du1 = 0, du2 = 0, du3 = 0;
Real dB1 = 0, dB2 = 0, dB3 = 0;

if (dir == 0)
{
B10 = 1.;
if (nmode == 0)
{ 
drho = 1.;
}
else if (nmode == 1)
{ 
omega = 2.35896379113i;
drho = 0.556500332363;
du = 0.742000443151;
du1 = -0.282334999306;
du2 = 0.0367010491491;
du3 = 0.0367010491491;
dB1 = -0.195509141461;
dB2 = 0.0977545707307;
dB3 = 0.0977545707307;
}
else if (nmode == 2)
{ 
omega = -3.44144232573i;
du2 = -0.339683110243;
du3 = 0.339683110243;
dB2 = 0.620173672946;
dB3 = -0.620173672946;
}
else
{ 
omega = 6.92915162882i;
drho = 0.481846076323;
du = 0.642461435098;
du1 = -0.0832240462505;
du2 = -0.224080007379;
du3 = -0.224080007379;
dB1 = 0.406380545676;
dB2 = -0.203190272838;
dB3 = -0.203190272838;
}
}
else
{
if (dir == 1)
{
B20 = 1.;
}
else if (dir == 2)
{
B30 = 1.;
}
else if (dir == 3)
{
B10 = 1.;
}

if (nmode == 0)
{ 
drho = 1.;
}
else if (nmode == 1)
{ 
omega = 2.41024185339i;
drho = 0.558104461559;
du = 0.744139282078;
if (dir == 1)
{
du2 = -0.277124827421;
du3 = 0.0630348927707;
dB2 = -0.164323721928;
dB3 = 0.164323721928;
}
else if (dir == 2)
{
du3 = -0.277124827421;
du1 = 0.0630348927707;
dB3 = -0.164323721928;
dB1 = 0.164323721928;
}
else if (dir == 3)
{
du1 = -0.277124827421;
du2 = 0.0630348927707;
dB1 = -0.164323721928;
dB2 = 0.164323721928;
}
}
else if (nmode == 2)
{ 
omega = 3.44144232573i;
if (dir == 1)
{
du1 = 0.480384461415;
dB1 = 0.877058019307;
}
else if (dir == 2)
{
du2 = 0.480384461415;
dB2 = 0.877058019307;
}
else if (dir == 3)
{
du3 = 0.480384461415;
dB3 = 0.877058019307;
}
}
else
{ 
omega = 5.53726217331i;
drho = 0.476395427447;
du = 0.635193903263;
if (dir == 1)
{
du2 = -0.102965815319;
du3 = -0.316873207561;
dB2 = 0.359559114174;
dB3 = -0.359559114174;
}
else if (dir == 2)
{
du3 = -0.102965815319;
du1 = -0.316873207561;
dB3 = 0.359559114174;
dB1 = -0.359559114174;
}
else if (dir == 3)
{
du1 = -0.102965815319;
du2 = -0.316873207561;
dB1 = 0.359559114174;
dB2 = -0.359559114174;
}
}
}

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
pmb->par_for("mhdmodes_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {
Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);

Real mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
rho(k, j, i) = rho0 + drho * mode;
u(k, j, i) = u0 + du * mode;
uvec(0, k, j, i) = u10 + du1 * mode;
uvec(1, k, j, i) = u20 + du2 * mode;
uvec(2, k, j, i) = u30 + du3 * mode;
B_P(0, k, j, i) = B10 + dB1 * mode;
B_P(1, k, j, i) = B20 + dB2 * mode;
B_P(2, k, j, i) = B30 + dB3 * mode;
}
);

if (nmode != 0 && one_period) {
pin->SetReal("parthenon/time", "tlim", 2. * M_PI / m::abs(omega.imag()));
}

return TaskStatus::complete;
}
