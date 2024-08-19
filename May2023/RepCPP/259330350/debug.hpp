
#pragma once

#include "decs.hpp"
#include "mpi.hpp"
#include "types.hpp"


TaskStatus CheckNaN(MeshData<Real> *md, int dir, IndexDomain domain=IndexDomain::interior);


TaskStatus CheckNegative(MeshData<Real> *md, IndexDomain domain=IndexDomain::interior);


int CountPFlags(MeshData<Real> *md, IndexDomain domain=IndexDomain::interior, int verbose=0);


int CountFFlags(MeshData<Real> *md, IndexDomain domain=IndexDomain::interior, int verbose=0);

KOKKOS_INLINE_FUNCTION void print_matrix(const std::string name, const double g[GR_DIM][GR_DIM], bool kill_on_nan=false)
{
printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", name.c_str(),
g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], g[1][1], g[1][2],
g[1][3], g[2][0], g[2][1], g[2][2], g[2][3], g[3][0], g[3][1],
g[3][2], g[3][3]);

if (kill_on_nan) {
DLOOP2 if (m::isnan(g[mu][nu])) exit(-1);
}
}
KOKKOS_INLINE_FUNCTION void print_vector(const std::string name, const double v[GR_DIM], bool kill_on_nan=false)
{
printf("%s: %g\t%g\t%g\t%g\n", name.c_str(), v[0], v[1], v[2], v[3]);

if (kill_on_nan) {
DLOOP2 if (m::isnan(v[nu])) exit(-1);
}
}
