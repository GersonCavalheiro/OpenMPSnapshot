

#ifndef LBT_UNIT_REYNOLDS_NUMBER_UNITTEST
#define LBT_UNIT_REYNOLDS_NUMBER_UNITTEST
#pragma once

#include <gtest/gtest.h>

#include "unit/detail/kinematic_viscosity.hpp"
#include "unit/detail/length.hpp"
#include "unit/detail/reynolds_number.hpp"
#include "unit/detail/velocity.hpp"


namespace lbt {
namespace unit {
namespace test {

TEST(ReynoldsNumberTest, constructor) {
constexpr long double expected_result {1234.56};
lbt::unit::ReynoldsNumber const reynolds_number {expected_result};
EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
}

TEST(ReynoldsNumberTest, constructorFromPhysicalUnits) {
lbt::unit::Velocity const velocity {1.7};
lbt::unit::Length const length {0.025};
lbt::unit::KinematicViscosity const kinematic_viscosity {0.0000016};
constexpr long double expected_result {26562.5};
lbt::unit::ReynoldsNumber const reynolds_number {velocity, length, kinematic_viscosity};
EXPECT_DOUBLE_EQ(reynolds_number.get(), expected_result);
}

TEST(ReynoldsNumberTest, conversionToLongDouble) {
constexpr long double expected_result {1234.56};
lbt::unit::ReynoldsNumber const reynolds_number {expected_result};
long double const result {reynolds_number};
EXPECT_DOUBLE_EQ(result, expected_result);
}

}
}
}

#endif 
