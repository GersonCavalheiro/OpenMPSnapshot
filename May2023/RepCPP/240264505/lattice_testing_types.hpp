

#ifndef LBT_LATTICE_TESTING_TYPES
#define LBT_LATTICE_TESTING_TYPES
#pragma once

#include <tuple>

#include <gtest/gtest.h>

#include "lattice/D2Q9.hpp"
#include "lattice/D3Q15.hpp"
#include "lattice/D3Q19.hpp"
#include "lattice/D3Q27.hpp"
#include "general/tuple_utilities.hpp"
#include "../testing_utilities/testing_utilities.hpp"


namespace lbt {
namespace test {

template <typename T> using LatticeTypes2D = std::tuple<lbt::lattice::D2Q9P10<T>,  lbt::lattice::D2Q9P12<T>>;
template <typename T> using LatticeTypes3D = std::tuple<lbt::lattice::D3Q15P16<T>, lbt::lattice::D3Q19P20<T>, 
lbt::lattice::D3Q27P28<T>, lbt::lattice::D3Q27PC<T>>;
template <typename T> using LatticeTypes = decltype(std::tuple_cat(std::declval<LatticeTypes2D<T>>(), 
std::declval<LatticeTypes3D<T>>()));

using LatticeFloatingTypes = std::tuple<double, float>;

using LatticeTestTypes2D = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes2D, LatticeFloatingTypes>>;
using LatticeTestTypes3D = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes3D, LatticeFloatingTypes>>;
using LatticeTestTypes = ToTestingTypes_t<lbt::CartesianProductApply_t<LatticeTypes, LatticeFloatingTypes>>;

}
}

#endif 
