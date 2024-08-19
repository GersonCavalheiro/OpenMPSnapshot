

#ifndef LBT_MATERIAL_STATE
#define LBT_MATERIAL_STATE
#pragma once

#include "../../unit/units.hpp"


namespace lbt {
namespace material {


class State {
public:

explicit constexpr State(lbt::unit::Pressure const pressure, lbt::unit::Temperature const temperature, lbt::unit::Density const density) noexcept
: pressure_{pressure}, temperature_{temperature}, density_{density} {
return;
}

State() = delete;
State(State const&) = default;
State& operator= (State const&) = default;
State(State&&) = default;
State& operator= (State&&) = default;


template <typename T>
constexpr T get() const noexcept = delete;

protected:
lbt::unit::Pressure pressure_;
lbt::unit::Temperature temperature_;
lbt::unit::Density density_;
};

template <>
constexpr lbt::unit::Pressure State::get() const noexcept {
return pressure_;
}

template <>
constexpr lbt::unit::Temperature State::get() const noexcept {
return temperature_;
}

template <>
constexpr lbt::unit::Density State::get() const noexcept {
return density_;
}

}
}

#endif 
