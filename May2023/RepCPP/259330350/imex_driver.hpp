
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

using namespace parthenon;


class ImexDriver : public MultiStageDriver {
public:

ImexDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageDriver(pin, papp, pm) {}


TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};
