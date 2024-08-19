

#ifndef LBT_INDEXING
#define LBT_INDEXING
#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

#include "../../general/type_definitions.hpp"


namespace lbt {


enum class Timestep: bool { Even = false, Odd = true };


inline constexpr Timestep operator! (Timestep const& ts) noexcept {
return (ts == Timestep::Even) ? Timestep::Odd : Timestep::Even;
}


inline std::ostream& operator << (std::ostream& os, Timestep const& ts) noexcept {
os << ((ts == Timestep::Even) ? "even time step" : "odd time step");
return os;
}


template <typename LT, std::int32_t NP>
class Indexing {
public:

constexpr Indexing(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept
: NX{NX}, NY{NY}, NZ{NZ} {
assert((LT::DIM == 2) ? (NZ == 1) : true); 
return;
}
Indexing() = delete;
Indexing(Indexing const&) = default;
Indexing(Indexing&&) = default;
Indexing& operator= (Indexing const&) = default;
Indexing& operator= (Indexing&&) = default;


LBT_FORCE_INLINE constexpr std::int64_t spatialToLinear(std::int32_t const x, std::int32_t const y, std::int32_t const z,
std::int32_t const n, std::int32_t const d, std::int32_t const p) const noexcept {
return (((static_cast<std::int64_t>(z)*NY + y)*NX + x)*NP + p)*LT::ND + n*LT::OFF + d;
}


constexpr std::tuple<std::int32_t,std::int32_t,std::int32_t,std::int32_t,std::int32_t,std::int32_t>
linearToSpatial(std::int64_t const index) const noexcept;

protected:
std::int32_t NX;
std::int32_t NY;
std::int32_t NZ;
};


template <typename LT, std::int32_t NP>
constexpr std::tuple<std::int32_t,std::int32_t,std::int32_t,std::int32_t,std::int32_t,std::int32_t>
Indexing<LT,NP>::linearToSpatial(std::int64_t const index) const noexcept {
std::int64_t factor {LT::ND*NP*NX*NY};
std::int64_t rest {index%factor};

std::int32_t const z {static_cast<std::int32_t>(index/factor)};

factor = LT::ND*NP*NX;
std::int32_t const y {static_cast<std::int32_t>(rest/factor)};
rest   = rest%factor;

factor = LT::ND*NP;
std::int32_t const x {static_cast<std::int32_t>(rest/factor)};
rest   = rest%factor;

factor = LT::ND;
std::int32_t const p {static_cast<std::int32_t>(rest/factor)};
rest   = rest%factor;

factor = LT::OFF;
std::int32_t const n {static_cast<std::int32_t>(rest/factor)};
rest   = rest%factor;

factor = LT::SPEEDS;
std::int32_t const d {static_cast<std::int32_t>(rest%factor)};

return std::make_tuple(x,y,z,n,d,p);
}

}

#endif 
