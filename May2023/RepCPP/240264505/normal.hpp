

#ifndef LBT_BOUNDARY_NORMAL
#define LBT_BOUNDARY_NORMAL
#pragma once

#include <cstdint>
#include <ostream>

#include "orientation.hpp"


namespace lbt {
namespace boundary{


template <Orientation O>
class Normal {
public:
static constexpr std::int32_t x =  0;
static constexpr std::int32_t y =  0;
static constexpr std::int32_t z =  0;
};

template <>
class Normal<Orientation::Left> {
public:
static constexpr std::int32_t x =  1; 
static constexpr std::int32_t y =  0;
static constexpr std::int32_t z =  0;
};

template <>
class Normal<Orientation::Right> {
public:
static constexpr std::int32_t x = -1;
static constexpr std::int32_t y =  0;
static constexpr std::int32_t z =  0;
};

template <>
class Normal<Orientation::Front> {
public:
static constexpr std::int32_t x =  0;
static constexpr std::int32_t y =  1;
static constexpr std::int32_t z =  0;
};

template <>
class Normal<Orientation::Back> {
public:
static constexpr std::int32_t x =  0;
static constexpr std::int32_t y = -1;
static constexpr std::int32_t z =  0;
};

template <>
class Normal<Orientation::Bottom> {
public:
static constexpr std::int32_t x =  0;
static constexpr std::int32_t y =  0;
static constexpr std::int32_t z =  1;
};

template <>
class Normal<Orientation::Top> {
public:
static constexpr std::int32_t x =  0;
static constexpr std::int32_t y =  0;
static constexpr std::int32_t z = -1;
};

}
}

#endif 
