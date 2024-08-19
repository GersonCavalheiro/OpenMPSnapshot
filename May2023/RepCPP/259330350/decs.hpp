
#pragma once



#include <map>
#include <memory>
#include <stdexcept>

#include "Kokkos_Core.hpp"

#if 1
namespace m = Kokkos::Experimental;
#else
namespace m = std;
#endif

#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "mesh/domain.hpp"


using parthenon::Real;
using GReal = double;

#define SMALL 1e-20

#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

#define NVEC 3
#define VLOOP for(int v = 0; v < NVEC; ++v)
#define VLOOP2 VLOOP for(int w = 0; w < NVEC; ++w)

#define PLOOP for(int ip=0; ip < nvar; ++ip)

#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

KOKKOS_INLINE_FUNCTION Loci loc_of(const int& dir)
{
switch (dir) {
case 0:
return Loci::center;
case parthenon::X1DIR:
return Loci::face1;
case parthenon::X2DIR:
return Loci::face2;
case parthenon::X3DIR:
return Loci::face3;
default:
return Loci::corner;
}
}
KOKKOS_INLINE_FUNCTION int dir_of(const Loci loc)
{
switch (loc) {
case Loci::center:
return 0;
case Loci::face1:
return parthenon::X1DIR;
case Loci::face2:
return parthenon::X2DIR;
case Loci::face3:
return parthenon::X3DIR;
default:
return -1;
}
}

using GridScalar = parthenon::ParArrayND<parthenon::Real>;
using GridVector = parthenon::ParArrayND<parthenon::Real>;
using GridVars = parthenon::ParArrayND<parthenon::Real>;  
using GridInt = parthenon::ParArrayND<int>;

using GeomScalar = parthenon::ParArrayND<parthenon::Real>;
using GeomVector = parthenon::ParArrayND<parthenon::Real>;
using GeomTensor2 = parthenon::ParArrayND<parthenon::Real>;
using GeomTensor3 = parthenon::ParArrayND<parthenon::Real>;

#define KOKKOS_LAMBDA_1D KOKKOS_LAMBDA (const int& i)
#define KOKKOS_LAMBDA_2D KOKKOS_LAMBDA (const int& j, const int& i)
#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_4D KOKKOS_LAMBDA (const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_5D KOKKOS_LAMBDA (const int& m, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VEC KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_1D KOKKOS_LAMBDA (const int& b, const int& i)
#define KOKKOS_LAMBDA_MESH_2D KOKKOS_LAMBDA (const int& b, const int& j, const int& i)
#define KOKKOS_LAMBDA_MESH_3D KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_4D KOKKOS_LAMBDA (const int& b, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_5D KOKKOS_LAMBDA (const int& b, const int& m, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_VARS KOKKOS_LAMBDA (const int& b, const int &p, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_VEC KOKKOS_LAMBDA (const int& b, const int &mu, const int &k, const int &j, const int &i)

#define KOKKOS_LAMBDA_1D_REDUCE KOKKOS_LAMBDA (const int &i, parthenon::Real &local_result)
#define KOKKOS_LAMBDA_2D_REDUCE KOKKOS_LAMBDA (const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_3D_REDUCE KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_3D_REDUCE_INT KOKKOS_LAMBDA (const int &k, const int &j, const int &i, int &local_result)
#define KOKKOS_LAMBDA_MESH_3D_REDUCE KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_MESH_3D_REDUCE_INT KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result)
#define KOKKOS_LAMBDA_MESH_4D_REDUCE KOKKOS_LAMBDA (const int &b, const int &v, const int &k, const int &j, const int &i, double &local_result)
