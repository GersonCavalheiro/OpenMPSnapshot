

#ifndef LBT_UNIT_DYNAMIC_VISCOSITY_LITERALS_UNITTEST
#define LBT_UNIT_DYNAMIC_VISCOSITY_LITERALS_UNITTEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "unit/detail/dynamic_viscosity.hpp"
#include "unit/detail/dynamic_viscosity_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
namespace literals {
namespace test {
using namespace lbt::literals;

using DynamicViscosityLiteralsHelper = UnitLiteralsHelper<lbt::unit::DynamicViscosity>;

TEST_P(DynamicViscosityLiteralsHelper, unitConversion) {
auto const [dynamic_viscosity, expected_result] = GetParam();
EXPECT_DOUBLE_EQ(dynamic_viscosity.get(), expected_result);
}

INSTANTIATE_TEST_SUITE_P(DynamicViscosityLiteralsTest, DynamicViscosityLiteralsHelper, ::testing::Values(
std::make_pair(2.4_Pas,        2.4L),
std::make_pair(1234.5_mPas,    1.2345L),
std::make_pair(2467242.3_uPas, 2.4672423L),
std::make_pair(7.8_P,          7.8e-1L),
std::make_pair(342.5_cP,       0.3425L)
)
);

}
}
}

#endif 
