
#pragma once

#include "decs.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

#include <memory>


namespace B_FluxCT {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


void UtoP(MeshData<Real> *md, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedMesh(MeshData<Real> *md) { UtoP(md); }
inline TaskStatus FillDerivedMeshTask(MeshData<Real> *md) { UtoP(md); return TaskStatus::complete; }
void UtoP(MeshBlockData<Real> *md, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedBlock(MeshBlockData<Real> *rc) { UtoP(rc); }
inline TaskStatus FillDerivedBlockTask(MeshBlockData<Real> *rc) { UtoP(rc); return TaskStatus::complete; }


void PtoU(MeshBlockData<Real> *md, IndexDomain domain=IndexDomain::interior, bool coarse=false);


TaskStatus FluxCT(MeshData<Real> *md);


TaskStatus FixPolarFlux(MeshData<Real> *md);


TaskStatus TransportB(MeshData<Real> *md);


double MaxDivB(MeshData<Real> *md);


double GlobalMaxDivB(MeshData<Real> *md);


void CleanupDivergence(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::interior, bool coarse=false);


TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md);
inline TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{ return PrintGlobalMaxDivB(md); }
TaskStatus PrintMaxBlockDivB(MeshBlockData<Real> *rc, bool prims, std::string tag);


void FillOutput(MeshBlock *pmb, ParameterInput *pin);

void CalcDivB(MeshData<Real> *md, std::string divb_field_name="divB");


template<typename Global>
KOKKOS_INLINE_FUNCTION double corner_div(const GRCoordinates& G, const Global& B_U, const int& b,
const int& k, const int& j, const int& i, const bool& do_3D, const bool& do_2D=true)
{
const double norm = (do_2D) ? ((do_3D) ? 0.25 : 0.5) : 1.;
double term1 = B_U(b, V1, k, j, i) - B_U(b, V1, k, j, i-1);
double term2 = 0.;
double term3 = 0.;
if (do_2D) {
term1 +=   B_U(b, V1, k, j-1, i) - B_U(b, V1, k, j-1, i-1);
term2 +=   B_U(b, V2, k, j, i)   + B_U(b, V2, k, j, i-1)
- B_U(b, V2, k, j-1, i) - B_U(b, V2, k, j-1, i-1);
term3 += 0.;
}
if (do_3D) {
term1 +=  B_U(b, V1, k-1, j, i)   + B_U(b, V1, k-1, j-1, i)
- B_U(b, V1, k-1, j, i-1) - B_U(b, V1, k-1, j-1, i-1);
term2 +=  B_U(b, V2, k-1, j, i)   + B_U(b, V2, k-1, j, i-1)
- B_U(b, V2, k-1, j-1, i) - B_U(b, V2, k-1, j-1, i-1);
term3 =   B_U(b, V3, k, j, i)     + B_U(b, V3, k, j-1, i)
+ B_U(b, V3, k, j, i-1)   + B_U(b, V3, k, j-1, i-1)
- B_U(b, V3, k-1, j, i)   - B_U(b, V3, k-1, j-1, i)
- B_U(b, V3, k-1, j, i-1) - B_U(b, V3, k-1, j-1, i-1);
}
return norm*term1/G.dx1v(i) + norm*term2/G.dx2v(j) + norm*term3/G.dx3v(k);
}


template<typename Global>
KOKKOS_INLINE_FUNCTION void center_grad(const GRCoordinates& G, const Global& P, const int& b,
const int& k, const int& j, const int& i, const bool& do_3D,
double& B1, double& B2, double& B3)
{
const double norm = (do_3D) ? 0.25 : 0.5;
double term1 =  P(b, 0, k, j+1, i+1) + P(b, 0, k, j, i+1)
- P(b, 0, k, j+1, i)   - P(b, 0, k, j, i);
double term2 =  P(b, 0, k, j+1, i+1) + P(b, 0, k, j+1, i)
- P(b, 0, k, j, i+1)   - P(b, 0, k, j, i);
double term3 = 0.;
if (do_3D) {
term1 += P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j, i+1)
- P(b, 0, k+1, j+1, i)   - P(b, 0, k+1, j, i);
term2 += P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j+1, i)
- P(b, 0, k+1, j, i+1)   - P(b, 0, k+1, j, i);
term3 =  P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j, i+1)
+ P(b, 0, k+1, j+1, i)   + P(b, 0, k+1, j, i)
- P(b, 0, k, j+1, i+1)   - P(b, 0, k, j, i+1)
- P(b, 0, k, j+1, i)     - P(b, 0, k, j, i);
}
B1 = norm*term1/G.dx1v(i);
B2 = norm*term2/G.dx2v(j);
B3 = norm*term3/G.dx3v(k);
}

}
