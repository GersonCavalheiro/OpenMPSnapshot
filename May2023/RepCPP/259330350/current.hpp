
#pragma once

#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "matrix.hpp"
#include "grmhd_functions.hpp"

namespace Current
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);


void FillOutput(MeshBlock *pmb, ParameterInput *pin);


TaskStatus CalculateCurrent(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const double& dt);


KOKKOS_INLINE_FUNCTION double get_gdet_Fcon(const GRCoordinates& G, GridVector uvec, GridVector B_P,
const int& mu, const int& nu, const int& k, const int& j, const int& i)
{
if (mu == nu) return 0.;

FourVectors Dtmp;
GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
double Fcon = 0.;
for (int kap = 0; kap < GR_DIM; kap++) {
for (int lam = 0; lam < GR_DIM; lam++) {
Fcon -= antisym(mu, nu, kap, lam) * Dtmp.ucov[kap] * Dtmp.bcov[lam];
}
}

return Fcon;
}

} 
