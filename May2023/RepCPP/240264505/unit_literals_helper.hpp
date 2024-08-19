

#ifndef LBT_UNIT_LITERALS_HELPER
#define LBT_UNIT_LITERALS_HELPER
#pragma once

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "unit/detail/unit_base.hpp"


namespace lbt {
namespace literals {
namespace test {


template <typename T, typename std::enable_if_t<lbt::unit::is_unit_v<T>>* = nullptr>
class UnitLiteralsHelper : public testing::Test, public ::testing::WithParamInterface<std::pair<T,long double>> {
};

}
}
}

#endif 
