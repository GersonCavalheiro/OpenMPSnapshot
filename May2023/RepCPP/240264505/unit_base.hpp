

#ifndef LBT_UNIT_UNIT_BASE
#define LBT_UNIT_UNIT_BASE
#pragma once

#include <iostream>
#include <type_traits>


namespace lbt {
namespace unit {

namespace detail {


template <typename T>
class UnitBase {
public:


friend constexpr T operator- (T const& a) noexcept {
return T{-a.value};
}


friend constexpr T operator+ (T const& a, T const& b) noexcept {
return T{a.value + b.value};
}


friend constexpr T operator- (T const& a, T const& b) noexcept {
return T{a.value - b.value};
}


friend constexpr long double operator/ (T const& t1, T const& t2) noexcept {
return t1.get()/t2.get();
}


friend constexpr T operator* (long double const c, T const& t) noexcept {
return T{c*t.value};
}
friend constexpr T operator* (T const& t, long double const c) noexcept {
return T{c*t.value};
}


friend constexpr T operator/ (T const& t, long double const c) noexcept {
return T{t.value/c};
}


friend std::ostream& operator<< (std::ostream& os, T const& t) noexcept {
os << t.value;
return os;
}


constexpr void set(long double const val) noexcept {
value = val;
return;
}


template <typename U = long double, typename std::enable_if_t<std::is_arithmetic_v<U>>* = nullptr>
constexpr auto get() const noexcept {
return static_cast<U>(value);
}

protected:

constexpr UnitBase(long double const value) noexcept 
: value{value} {
return;
}
UnitBase(UnitBase const&) = default;
UnitBase& operator= (UnitBase const&) = default;
UnitBase(UnitBase&&) = default;
UnitBase& operator= (UnitBase&&) = default;

template <typename U> friend class UnitBase; 

long double value; 
};

constexpr std::false_type is_unit_impl(...) noexcept;
template <typename T>
constexpr std::true_type is_unit_impl(UnitBase<T> const volatile&) noexcept;
}


template <typename T>
using is_unit = decltype(lbt::unit::detail::is_unit_impl(std::declval<T&>()));
template <typename T>
static constexpr bool is_unit_v = is_unit<T>::value;

}
}

#endif 
