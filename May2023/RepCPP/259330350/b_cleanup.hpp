
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"
#include "types.hpp"

using namespace parthenon;


namespace B_Cleanup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


void CleanupDivergence(std::shared_ptr<MeshData<Real>>& md);




TaskStatus CornerLaplacian(MeshData<Real>* md, const std::string& p_var, const std::string& lap_var);


TaskStatus ApplyP(MeshData<Real> *msolve, MeshData<Real> *md);

} 
