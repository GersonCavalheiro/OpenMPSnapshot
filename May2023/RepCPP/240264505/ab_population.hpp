

#ifndef LBT_AB_POPULATION
#define LBT_AB_POPULATION
#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

#include "../general/type_definitions.hpp"
#include "indexing/indexing.hpp"


namespace lbt {


template <class LT, std::int32_t NP = 1>
class AbPopulation: public Indexing<LT,NP> {
public:

AbPopulation(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
: Indexing<LT,NP>{NX, NY, NZ}, A(static_cast<std::int64_t>(NZ)*NY*NX*NP*LT::ND), B(static_cast<std::int64_t>(NZ)*NY*NX*NP*LT::ND) {
assert((LT::DIM == 2) ? (NZ == 1) : true); 
return;
}
AbPopulation() = delete;
AbPopulation(AbPopulation const&) = default;
AbPopulation& operator= (AbPopulation const&) = default;
AbPopulation(AbPopulation&&) = default;
AbPopulation& operator= (AbPopulation&&) = default;


template <Timestep TS>
LBT_FORCE_INLINE auto& read(std::int32_t const x, std::int32_t const y, std::int32_t const z,
std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) noexcept {
if constexpr (TS == Timestep::Even) {
return A[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
} else {
return B[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
}
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& read(std::int32_t const x, std::int32_t const y, std::int32_t const z,
std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) const noexcept {
if constexpr (TS == Timestep::Even) {
return A[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
} else {
return B[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
}
}


template <Timestep TS>
LBT_FORCE_INLINE auto& write(std::int32_t const x, std::int32_t const y, std::int32_t const z,
std::int32_t const n, std::int32_t const d, std::int32_t const p = 0) noexcept {
if constexpr (TS == Timestep::Even) {
return B[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
} else {
return A[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
}
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& write(std::int32_t const x, std::int32_t const y, std::int32_t const z,
std::int32_t const n, std::int32_t const d, std::int32_t const p) const noexcept {
if constexpr (TS == Timestep::Even) {
return B[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
} else {
return A[Indexing<LT,NP>::spatialToLinear(x,y,z,n,d,p)];
}
}

protected:
using T = typename LT::type;
LBT_ALIGN lbt::HeapArray<T> A;
LBT_ALIGN lbt::HeapArray<T> B;
};

}

#endif 
