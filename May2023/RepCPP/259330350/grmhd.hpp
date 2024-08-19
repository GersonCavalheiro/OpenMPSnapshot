
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

using namespace parthenon;


namespace GRMHD {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedBlock(MeshBlockData<Real> *rc) { UtoP(rc); }
inline TaskStatus FillDerivedBlockTask(MeshBlockData<Real> *rc) { UtoP(rc); return TaskStatus::complete; }


TaskStatus FixUtoP(MeshBlockData<Real> *rc);

void PostUtoP(MeshBlockData<Real> *rc);


TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);


Real EstimateTimestep(MeshBlockData<Real> *rc);

Real EstimateRadiativeTimestep(MeshBlockData<Real> *rc);


AmrTag CheckRefinement(MeshBlockData<Real> *rc);


void FillOutput(MeshBlock *pmb, ParameterInput *pin);


TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);
}
