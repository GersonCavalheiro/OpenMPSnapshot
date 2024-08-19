

#ifndef LBT_AA_POPULATION
#define LBT_AA_POPULATION
#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

#include "../general/type_definitions.hpp"
#include "indexing/aa_pattern.hpp"
#include "indexing/indexing.hpp"


namespace lbt {


template <class LT, std::int32_t NP = 1>
class AaPopulation: public AaPattern<LT,NP> {
public:

AaPopulation(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
: AaPattern<LT,NP>{NX, NY, NZ}, A(static_cast<std::int64_t>(NZ)*NY*NX*NP*LT::ND) {
assert((LT::DIM == 2) ? (NZ == 1) : true); 
return;
}
AaPopulation() = delete;
AaPopulation(AaPopulation const&) = default;
AaPopulation& operator= (AaPopulation const&) = default;
AaPopulation(AaPopulation&&) = default;
AaPopulation& operator= (AaPopulation&&) = default;


template <Timestep TS>
LBT_FORCE_INLINE auto& read(std::array<std::int32_t,3> const &x,
std::array<std::int32_t,3> const &y,
std::array<std::int32_t,3> const &z,
std::int32_t               const n,
std::int32_t               const d,
std::int32_t               const p = 0) noexcept {
return A[AaPattern<LT,NP>::template indexRead<TS>(x,y,z,n,d,p)];
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& read(std::array<std::int32_t,3> const &x,
std::array<std::int32_t,3> const &y,
std::array<std::int32_t,3> const &z,
std::int32_t               const n,
std::int32_t               const d,
std::int32_t               const p = 0) const noexcept {
return A[AaPattern<LT,NP>::template indexRead<TS>(x,y,z,n,d,p)];
}
template <Timestep TS>
LBT_FORCE_INLINE auto& read(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p = 0) noexcept {
return A[AaPattern<LT,NP>::template indexRead<TS>(x,y,z,n,d,p)];
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& read(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p = 0) const noexcept {
return A[AaPattern<LT,NP>::template indexRead<TS>(x,y,z,n,d,p)];
}


template <Timestep TS>
LBT_FORCE_INLINE auto& write(std::array<std::int32_t,3> const &x, 
std::array<std::int32_t,3> const &y,
std::array<std::int32_t,3> const &z,
std::int32_t               const n,
std::int32_t               const d,
std::int32_t               const p = 0) noexcept {
return A[AaPattern<LT,NP>::template indexWrite<TS>(x,y,z,n,d,p)];
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& write(std::array<std::int32_t,3> const &x,
std::array<std::int32_t,3> const &y,
std::array<std::int32_t,3> const &z,
std::int32_t               const n,
std::int32_t               const d,
std::int32_t               const p = 0) const noexcept {
return A[AaPattern<LT,NP>::template indexWrite<TS>(x,y,z,n,d,p)];
}
template <Timestep TS>
LBT_FORCE_INLINE auto& write(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p = 0) noexcept {
return A[AaPattern<LT,NP>::template indexWrite<TS>(x,y,z,n,d,p)];
}

template <Timestep TS>
LBT_FORCE_INLINE auto const& write(std::int32_t const x,
std::int32_t const y,
std::int32_t const z,
std::int32_t const n,
std::int32_t const d,
std::int32_t const p = 0) const noexcept {
return A[AaPattern<LT,NP>::template indexWrite<TS>(x,y,z,n,d,p)];
}


protected:
using T = typename LT::type;
LBT_ALIGN lbt::HeapArray<T> A;
};

}

#endif 
