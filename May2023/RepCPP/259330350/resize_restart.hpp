#pragma once

#include "decs.hpp"
#include "types.hpp"


void ReadIharmRestartHeader(std::string fname, std::unique_ptr<parthenon::ParameterInput>& pin);


TaskStatus ReadIharmRestart(MeshBlockData<Real> *rc, parthenon::ParameterInput *pin);
