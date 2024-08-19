

#ifndef LBT_BASE_SIMULATION
#define LBT_BASE_SIMULATION
#pragma once

#include "nlohmann/json.hpp"


namespace lbt {


class BaseSimulation {
using json = nlohmann::json;

protected:

constexpr BaseSimulation() noexcept {
return;
}

public:

virtual void run() noexcept = 0;


virtual json toJson() const noexcept = 0;
};

}

#endif 