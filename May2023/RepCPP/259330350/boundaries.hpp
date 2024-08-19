
#pragma once

#include "decs.hpp"

#include "bondi.hpp"
#include "grmhd_functions.hpp"


namespace KBoundaries {


void InnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void InnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);


TaskStatus FixFlux(MeshData<Real> *rc);


TaskID AddBoundarySync(TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1);


void SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds=true);


KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, const VariablePack<Real>& P, const IndexDomain domain,
const int& index_u1, const int& k, const int& j, const int& i)
{
Real uvec[NVEC], ucon[GR_DIM];
VLOOP uvec[v] = P(index_u1 + v, k, j, i);
GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);

if (((ucon[1] > 0.) && (domain == IndexDomain::inner_x1)) ||
((ucon[1] < 0.) && (domain == IndexDomain::outer_x1)))
{
double gamma = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);
VLOOP uvec[v] /= gamma;

Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
uvec[V1] = beta1 / alpha;

Real vsq = G.gcov(Loci::center, j, i, 1, 1) * uvec[V1] * uvec[V1] +
G.gcov(Loci::center, j, i, 2, 2) * uvec[V2] * uvec[V2] +
G.gcov(Loci::center, j, i, 3, 3) * uvec[V3] * uvec[V3] +
2. * (G.gcov(Loci::center, j, i, 1, 2) * uvec[V1] * uvec[V2] +
G.gcov(Loci::center, j, i, 1, 3) * uvec[V1] * uvec[V3] +
G.gcov(Loci::center, j, i, 2, 3) * uvec[V2] * uvec[V3]);

clip(vsq, 1.e-13, 1. - 1./(50.*50.));

gamma = 1./m::sqrt(1. - vsq);

VLOOP uvec[v] *= gamma;
VLOOP P(index_u1 + v, k, j, i) = uvec[v];
}
}

}
