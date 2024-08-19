

#ifndef LBT_CONTINUUM_BASE_UNITTEST
#define LBT_CONTINUUM_BASE_UNITTEST
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "constexpr_math/constexpr_math.hpp"
#include "../testing_utilities/testing_utilities.hpp"


namespace lbt {
namespace test{


template <typename C>
class ContinuumBaseTest {
using T = TemplateDataType_t<C>;
public:

ContinuumBaseTest(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ, 
std::filesystem::path const& output_path) noexcept
: NX{NX}, NY{NX}, NZ{NZ}, output_path{output_path}, c{NX, NY, NZ, output_path} {
return;
}


bool testSetAndGetPressure() noexcept {
using namespace std::placeholders;
auto setter {std::bind(&C::setP, std::ref(c), _1, _2, _3, _4)};
auto getter {std::bind(&C::getP, std::ref(c), _1, _2, _3)};
return testSetAndGetHelper(setter, getter);
}
bool testSetAndGetVelocityX() noexcept {
using namespace std::placeholders;
auto setter {std::bind(&C::setU, std::ref(c), _1, _2, _3, _4)};
auto getter {std::bind(&C::getU, std::ref(c), _1, _2, _3)};
return testSetAndGetHelper(setter, getter);
}
bool testSetAndGetVelocityY() noexcept {
using namespace std::placeholders;
auto setter {std::bind(&C::setV, std::ref(c), _1, _2, _3, _4)};
auto getter {std::bind(&C::getV, std::ref(c), _1, _2, _3)};
return testSetAndGetHelper(setter, getter);
}
bool testSetAndGetVelocityZ() noexcept {
using namespace std::placeholders;
auto setter {std::bind(&C::setW, std::ref(c), _1, _2, _3, _4)};
auto getter {std::bind(&C::getW, std::ref(c), _1, _2, _3)};
return testSetAndGetHelper(setter, getter);
}
protected:

bool testSetAndGetHelper(std::function<void(std::int32_t const, std::int32_t const, std::int32_t const, T const)> setter, 
std::function<T(std::int32_t const, std::int32_t const, std::int32_t const)> getter) noexcept {
bool is_success {true};
std::int64_t i {0};
for (std::int32_t x = 0; x < NX; ++x) {
for (std::int32_t y = 0; y < NY; ++y) {
for (std::int32_t z = 0; z < NZ; ++z) {
setter(x,y,z,static_cast<T>(i));
++i;
}
}
}
i = 0;
for (std::int32_t x = 0; x < NX; ++x) {
for (std::int32_t y = 0; y < NY; ++y) {
for (std::int32_t z = 0; z < NZ; ++z) {
is_success &= lbt::cem::isAlmostEqualEpsAbs(getter(x,y,z), static_cast<T>(i));
++i;
}
}
}
return is_success;
}

std::int32_t NX;
std::int32_t NY;
std::int32_t NZ;
std::filesystem::path output_path;
C c;
};
}
}

#endif 
