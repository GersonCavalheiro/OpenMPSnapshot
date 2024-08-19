

#ifndef LBT_CEM_IS_ALMOST_EQUAL_ULPS
#define LBT_CEM_IS_ALMOST_EQUAL_ULPS
#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace lbt {
namespace cem {

namespace detail {
template <typename T, std::size_t = sizeof(T)>
std::true_type is_complete(T*);

std::false_type is_complete(...);
}

template <typename T>
using is_complete = decltype(detail::is_complete(std::declval<T*>()));

template <typename T>
static constexpr bool is_complete_v = is_complete<T>::value;

namespace detail {


template <std::size_t N>
class UIntEquiv {
protected:
UIntEquiv() = delete;
UIntEquiv(UIntEquiv const&) = delete;
UIntEquiv(UIntEquiv&&) = delete;
UIntEquiv& operator= (UIntEquiv const&) = delete;
UIntEquiv& operator= (UIntEquiv&&) = delete;

template<std::size_t M, typename std::enable_if_t<(M==sizeof(std::uint8_t))>* = nullptr>
static constexpr std::uint8_t determineUIntEquivalent() noexcept;

template<std::size_t M, typename std::enable_if_t<(M==sizeof(std::uint16_t))>* = nullptr>
static constexpr std::uint16_t determineUIntEquivalent() noexcept;

template<std::size_t M, typename std::enable_if_t<(M==sizeof(std::uint32_t))>* = nullptr>
static constexpr std::uint32_t determineUIntEquivalent() noexcept;

template<std::size_t M, typename std::enable_if_t<(M==sizeof(std::uint64_t))>* = nullptr>
static constexpr std::uint64_t determineUIntEquivalent() noexcept;

public:
using type = decltype(determineUIntEquivalent<N>());
};

template <std::size_t N>
using UIntEquiv_t = typename UIntEquiv<N>::type;


template <typename T>
class FloatTrait;

template <typename T>
requires std::is_floating_point_v<T> && std::numeric_limits<T>::is_iec559 && (std::endian::native == std::endian::little)
class FloatTrait<T> {
public:
static constexpr std::size_t number_of_bytes {sizeof(T)};
static constexpr std::size_t number_of_bits {number_of_bytes*std::numeric_limits<std::uint8_t>::digits};
using Bytes = UIntEquiv_t<number_of_bytes>;

static constexpr std::size_t number_of_sign_bits {1};
static constexpr std::size_t number_of_fraction_bits {std::numeric_limits<T>::digits-1};
static constexpr std::size_t number_of_exponent_bits {number_of_bits - number_of_sign_bits - number_of_fraction_bits};

static constexpr Bytes sign_mask {Bytes{1} << (number_of_bits - 1)};
static constexpr Bytes fraction_mask {~Bytes{0} >> (number_of_exponent_bits + 1)};
static constexpr Bytes exponent_mask {~(sign_mask | fraction_mask)};


static constexpr bool isNan(T const t) noexcept {
auto const bytes {std::bit_cast<Bytes>(t)};
auto const exponent_bytes {extractExponent(bytes)};
auto const fraction_bytes {extractFraction(bytes)};
return (exponent_bytes == exponent_mask) && (fraction_bytes != 0);
}


static constexpr bool isPosInf(T const t) noexcept {
return isPos(t) && isInf(t);
}


static constexpr bool isNegInf(T const t) noexcept {
return isNeg(t) && isInf(t);
}


static constexpr bool isNeg(T const t) noexcept {
auto const bytes {std::bit_cast<Bytes>(t)};
auto const sign_bytes {extractSign(bytes)};
return sign_bytes != 0;
}


static constexpr bool isPos(T const t) noexcept {
auto const bytes {std::bit_cast<Bytes>(t)};
auto const sign_bytes {extractSign(bytes)};
return sign_bytes == 0;
}


static constexpr bool isInf(T const t) noexcept {
auto const bytes {std::bit_cast<Bytes>(t)};
auto const exponent_bytes {extractExponent(bytes)};
auto const fraction_bytes {extractFraction(bytes)};
return (exponent_bytes == exponent_mask) && (fraction_bytes == 0);
}


static constexpr Bytes extractSign(Bytes const bytes) noexcept {
return bytes & sign_mask;
}


static constexpr Bytes extractExponent(Bytes const bytes) noexcept {
return bytes & exponent_mask;
}


static constexpr Bytes extractFraction(Bytes const bytes) noexcept {
return bytes & fraction_mask;
}

protected:
FloatTrait() = delete;
FloatTrait(FloatTrait const&) = delete;
FloatTrait(FloatTrait&&) = delete;
FloatTrait& operator= (FloatTrait const&) = delete;
FloatTrait& operator= (FloatTrait&&) = delete;
};


template <typename T>
requires is_complete_v<FloatTrait<T>> && is_complete_v<UIntEquiv_t<sizeof(T)>>
class FloatView {
public:
using Trait = FloatTrait<T>;
using Bytes = UIntEquiv_t<sizeof(T)>;


explicit constexpr FloatView(T const t) noexcept
: value{t} {
return;
}
FloatView() = default;
FloatView(FloatView const&) = default;
FloatView(FloatView&&) = default;
FloatView& operator= (FloatView const&) = default;
FloatView& operator= (FloatView&&) = default;


constexpr bool isAlmostEqual(FloatView const rhs, std::uint8_t const max_distance = 4) const noexcept {
if (Trait::isNan(value) || Trait::isNan(rhs.value)) {
return false;
} else if (Trait::isNegInf(value) != Trait::isNegInf(rhs.value)) {
return false;
} else if (Trait::isPosInf(value) != Trait::isPosInf(rhs.value)) {
return false;
}
return computeDistance(value, rhs.value) <= max_distance;
}

protected:
T value;


static constexpr Bytes signMagnitudeToBiased(T const t) noexcept {
auto const b {std::bit_cast<Bytes>(t)};
if (Trait::isNeg(t)) {
return ~b + Bytes{1};
} else {
return Trait::sign_mask | b;
}
}


static constexpr Bytes computeDistance(T const a, T const b) noexcept {
auto const biased_a = signMagnitudeToBiased(a);
auto const biased_b = signMagnitudeToBiased(b);
return (biased_a >= biased_b) ? (biased_a - biased_b) : (biased_b - biased_a);
}
};

}


template <typename T>
constexpr bool isAlmostEqualUlps(T const lhs, T const rhs, std::uint8_t const max_distance = 4) noexcept {
detail::FloatView<T> const a {lhs};
detail::FloatView<T> const b {rhs};
return a.isAlmostEqual(b, max_distance);
}

}
}

#endif 
