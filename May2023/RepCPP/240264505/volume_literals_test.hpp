

#ifndef LBT_UNIT_VOLUME_LITERALS_UNITTEST
#define LBT_UNIT_VOLUME_LITERALS_UNITTEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "unit/detail/volume.hpp"
#include "unit/detail/volume_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
namespace literals {
namespace test {
using namespace lbt::literals;

using VolumeLiteralsHelper = UnitLiteralsHelper<lbt::unit::Volume>;

TEST_P(VolumeLiteralsHelper, unitConversion) {
auto const [volume, expected_result] = GetParam();
EXPECT_DOUBLE_EQ(volume.get(), expected_result);
}

INSTANTIATE_TEST_SUITE_P(VolumeLiteralsTest, VolumeLiteralsHelper, ::testing::Values(
std::make_pair(0.3_km3, 3.0e+8L),
std::make_pair(8.6_m3,  8.6L),
std::make_pair(1.2_cm3, 1.2e-6L),
std::make_pair(9.8_mm3, 9.8e-9L)
)
);

}
}
}

#endif 
