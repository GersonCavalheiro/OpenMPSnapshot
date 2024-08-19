
#pragma once

#include "decs.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

#include <memory>

using namespace parthenon;


namespace B_CD {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerived(MeshBlockData<Real> *rc) { UtoP(rc); }


TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);


Real MaxDivB(MeshData<Real> *md);


TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);


void FillOutput(MeshBlock *pmb, ParameterInput *pin);

}
