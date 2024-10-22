

#ifndef LBT_UNIT_AMOUNT_OF_SUBSTANCE_LITERALS_UNITTEST
#define LBT_UNIT_AMOUNT_OF_SUBSTANCE_LITERALS_UNITTEST
#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "unit/detail/amount_of_substance.hpp"
#include "unit/detail/amount_of_substance_literals.hpp"
#include "unit_literals_helper.hpp"


namespace lbt {
namespace literals {
namespace test {
using namespace lbt::literals;

using AmountOfSubstanceLiteralsHelper = UnitLiteralsHelper<lbt::unit::AmountOfSubstance>;

TEST_P(AmountOfSubstanceLiteralsHelper, unitConversion) {
auto const [amount_of_substance, expected_result] = GetParam();
EXPECT_DOUBLE_EQ(amount_of_substance.get(), expected_result);
}

INSTANTIATE_TEST_SUITE_P(AmountOfSubstanceLiteralsTest, AmountOfSubstanceLiteralsHelper, ::testing::Values(
std::make_pair(9.9_mol,  9.9L),
std::make_pair(0.7_kmol, 0.7e+3L)
)
);

}
}
}

#endif 
