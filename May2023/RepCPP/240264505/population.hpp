

#ifndef LBT_POPULATION
#define LBT_POPULATION
#pragma once

#include <cstdint>

#include "aa_population.hpp"


namespace lbt {
template <class LT, std::int32_t NP = 1>
using Population = AaPopulation<LT, NP>;
}

#endif 
