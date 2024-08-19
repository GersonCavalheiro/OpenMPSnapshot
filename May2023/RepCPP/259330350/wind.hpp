
#pragma once

#include "grmhd_functions.hpp"

#include <parthenon/parthenon.hpp>

namespace Wind {


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);


TaskStatus AddSource(MeshData<Real> *mdudt);

}
