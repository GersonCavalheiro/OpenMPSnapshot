#pragma once

#include <random>

namespace KMeans {

auto getGenerator(int seed = 123) {
static std::random_device rdev;
static std::mt19937_64 gen(rdev());
gen.seed(seed);
return gen;
}

} 
