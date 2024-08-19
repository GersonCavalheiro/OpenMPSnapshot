

#ifndef LBT_D2Q9
#define LBT_D2Q9
#pragma once

#include <cstdint>
#include <type_traits>

#include "../constexpr_math/constexpr_math.hpp"
#include "../general/type_definitions.hpp"


namespace lbt {
namespace lattice {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
class D2Q9P10 final {
public:
using type = T;

static constexpr std::int32_t    DIM =  2;
static constexpr std::int32_t SPEEDS =  9;
static constexpr std::int32_t HSPEED = (SPEEDS + 1)/2;

static constexpr std::int32_t PAD = 1;
static constexpr std::int32_t  ND = SPEEDS + PAD;
static constexpr std::int32_t OFF = ND/2;

LBT_ALIGN static constexpr lbt::StackArray<T, ND> DX =
{ 0,  1,  0,  1, -1,   
0, -1,  0, -1,  1 }; 
LBT_ALIGN static constexpr lbt::StackArray<T, ND> DY =
{ 0,  0,  1,  1,  1,
0,  0, -1, -1, -1 };
LBT_ALIGN static constexpr lbt::StackArray<T, ND> DZ =
{ 0,  0,  0,  0,  0,
0,  0,  0,  0,  0 };

LBT_ALIGN static constexpr lbt::StackArray<T, ND> W =
{ 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,
4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0 };

LBT_ALIGN static constexpr lbt::StackArray<T, ND> MASK =
{ 1, 1, 1, 1, 1,
0, 1, 1, 1, 1 };

static constexpr T CS = 1.0/cem::sqrt(3.0);
};

template<typename T> using D2Q9 = D2Q9P10<T>;



template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
class D2Q9P12 final {
public:
using type = T;

static constexpr std::int32_t    DIM =  2;
static constexpr std::int32_t SPEEDS =  9;
static constexpr std::int32_t HSPEED = (SPEEDS + 1)/2;

static constexpr std::int32_t PAD = 3;
static constexpr std::int32_t  ND = SPEEDS + PAD;
static constexpr std::int32_t OFF = ND/2;

LBT_ALIGN static constexpr lbt::StackArray<T, ND> DX =
{ 0,  1,  0,  1, -1,  0,   
0, -1,  0, -1,  1,  0 }; 
LBT_ALIGN static constexpr lbt::StackArray<T, ND> DY =
{ 0,  0,  1,  1,  1,  0, 
0,  0, -1, -1, -1,  0 };
LBT_ALIGN static constexpr lbt::StackArray<T, ND> DZ =
{ 0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0 };

LBT_ALIGN static constexpr lbt::StackArray<T, ND> W =
{ 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,  0.0, 
4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0,  0.0 };

LBT_ALIGN static constexpr lbt::StackArray<T, ND> MASK =
{ 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 0 };

static constexpr T CS = 1.0/cem::sqrt(3.0);
};

}
}

#endif 
