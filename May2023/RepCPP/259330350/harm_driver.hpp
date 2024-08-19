
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

#include "types.hpp"

using namespace parthenon;


class HARMDriver : public MultiStageDriver {
public:

HARMDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageDriver(pin, papp, pm) {}


TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};