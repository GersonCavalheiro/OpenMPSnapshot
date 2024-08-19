

#ifndef LBT_AA_PATTERN
#define LBT_AA_PATTERN
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "../../general/type_definitions.hpp"
#include "indexing.hpp"


namespace lbt {


template <typename LT, std::int32_t NP>
class AaPattern: public Indexing<LT,NP> {
public:
constexpr AaPattern(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
: Indexing<LT,NP>{NX,NY,NZ} {
assert((LT::DIM == 2) ? (NZ == 1) : true); 
return;
}
AaPattern() = delete;
AaPattern(AaPattern const&) = default;
AaPattern& operator= (AaPattern const&) = default;
AaPattern(AaPattern&&) = default;
AaPattern& operator= (AaPattern&&) = default;


template <Timestep TS>
LBT_FORCE_INLINE static constexpr std::int32_t oddEven(std::int32_t const odd_index, std::int32_t const even_index) noexcept {
return (TS == Timestep::Odd) ? odd_index : even_index;
}


template <Timestep TS>
LBT_FORCE_INLINE constexpr std::int64_t indexRead(lbt::StackArray<std::int32_t,3> const& x,
lbt::StackArray<std::int32_t,3> const& y,
lbt::StackArray<std::int32_t,3> const& z,
std::int32_t                    const n,
std::int32_t                    const d,
std::int32_t                    const p) const noexcept {
return Indexing<LT,NP>::spatialToLinear(x[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DX[(!n)*LT::OFF+d]), 0)],
y[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DY[(!n)*LT::OFF+d]), 0)],
z[1 + oddEven<TS>(static_cast<std::int32_t>(LT::DZ[(!n)*LT::OFF+d]), 0)],
oddEven<TS>(n, !n),
d, p);
}
template <Timestep TS>
LBT_FORCE_INLINE constexpr std::int64_t indexRead(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p) const noexcept {
if constexpr (TS == Timestep::Odd) {
std::int32_t const xn = (Indexing<LT,NP>::NX + x + static_cast<std::int32_t>(LT::DX[(!n)*LT::OFF+d])) % Indexing<LT,NP>::NX;
std::int32_t const yn = (Indexing<LT,NP>::NY + y + static_cast<std::int32_t>(LT::DY[(!n)*LT::OFF+d])) % Indexing<LT,NP>::NY;
std::int32_t const zn = (Indexing<LT,NP>::NZ + z + static_cast<std::int32_t>(LT::DZ[(!n)*LT::OFF+d])) % Indexing<LT,NP>::NZ;
return Indexing<LT,NP>::spatialToLinear(xn,yn,zn,n,d,p);
} else {
return Indexing<LT,NP>::spatialToLinear(x,y,z,!n,d,p);
}
}


template <Timestep TS>
LBT_FORCE_INLINE constexpr std::int64_t indexWrite(lbt::StackArray<std::int32_t,3> const& x,
lbt::StackArray<std::int32_t,3> const& y,
lbt::StackArray<std::int32_t,3> const& z,
std::int32_t                    const n,
std::int32_t                    const d,
std::int32_t                    const p) const noexcept {
return Indexing<LT,NP>::spatialToLinear(x[1 - oddEven<TS>(0, static_cast<std::int32_t>(LT::DX[n*LT::OFF+d]))],
y[1 - oddEven<TS>(0, static_cast<std::int32_t>(LT::DY[n*LT::OFF+d]))],
z[1 - oddEven<TS>(0, static_cast<std::int32_t>(LT::DZ[n*LT::OFF+d]))],
oddEven<TS>(!n, n),
d, p);
}
template <Timestep TS>
LBT_FORCE_INLINE constexpr std::int64_t indexWrite(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p) const noexcept {
if constexpr (TS == Timestep::Odd) {
return Indexing<LT,NP>::spatialToLinear(x,y,z,!n,d,p);
} else {
std::int32_t const xn = (Indexing<LT,NP>::NX + x - static_cast<std::int32_t>(LT::DX[n*LT::OFF+d])) % Indexing<LT,NP>::NX;
std::int32_t const yn = (Indexing<LT,NP>::NY + y - static_cast<std::int32_t>(LT::DY[n*LT::OFF+d])) % Indexing<LT,NP>::NY;
std::int32_t const zn = (Indexing<LT,NP>::NZ + z - static_cast<std::int32_t>(LT::DZ[n*LT::OFF+d])) % Indexing<LT,NP>::NZ;
return Indexing<LT,NP>::spatialToLinear(xn,yn,zn,n,d,p);
}
}
};

}

#endif 
