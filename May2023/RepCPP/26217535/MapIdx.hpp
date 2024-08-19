

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/vec/Vec.hpp>

#include <type_traits>

namespace alpaka
{
namespace detail
{
template<std::size_t TidxDimOut, std::size_t TidxDimIn, typename TSfinae = void>
struct MapIdx;
template<std::size_t TidxDim>
struct MapIdx<TidxDim, TidxDim>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdx(
Vec<DimInt<TidxDim>, TElem> const& idx,
[[maybe_unused]] Vec<DimInt<TidxDim>, TElem> const& extent) -> Vec<DimInt<TidxDim>, TElem>
{
return idx;
}
};
template<std::size_t TidxDimOut>
struct MapIdx<TidxDimOut, 1u, std::enable_if_t<(TidxDimOut > 1u)>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdx(
Vec<DimInt<1u>, TElem> const& idx,
Vec<DimInt<TidxDimOut>, TElem> const& extent) -> Vec<DimInt<TidxDimOut>, TElem>
{
auto idxNd = Vec<DimInt<TidxDimOut>, TElem>::all(0u);

constexpr std::size_t lastIdx(TidxDimOut - 1u);

idxNd[lastIdx] = static_cast<TElem>(idx[0u] % extent[lastIdx]);

TElem hyperPlanesBefore = extent[lastIdx];
for(std::size_t r(1u); r < lastIdx; ++r)
{
std::size_t const d = lastIdx - r;
idxNd[d] = static_cast<TElem>(idx[0u] / hyperPlanesBefore % extent[d]);
hyperPlanesBefore *= extent[d];
}

idxNd[0u] = static_cast<TElem>(idx[0u] / hyperPlanesBefore);

return idxNd;
}
};
template<std::size_t TidxDimIn>
struct MapIdx<1u, TidxDimIn, std::enable_if_t<(TidxDimIn > 1u)>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdx(
Vec<DimInt<TidxDimIn>, TElem> const& idx,
Vec<DimInt<TidxDimIn>, TElem> const& extent) -> Vec<DimInt<1u>, TElem>
{
TElem idx1d(idx[0u]);
for(std::size_t d(1u); d < TidxDimIn; ++d)
{
idx1d = static_cast<TElem>(idx1d * extent[d] + idx[d]);
}
return {idx1d};
}
};

template<std::size_t TidxDimOut>
struct MapIdx<TidxDimOut, 0u>
{
template<typename TElem, std::size_t TidxDimExtents>
ALPAKA_FN_HOST_ACC static auto mapIdx(
Vec<DimInt<0u>, TElem> const&,
Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<TidxDimOut>, TElem>
{
return Vec<DimInt<TidxDimOut>, TElem>::zeros();
}
};

template<std::size_t TidxDimIn>
struct MapIdx<0u, TidxDimIn>
{
template<typename TElem, std::size_t TidxDimExtents>
ALPAKA_FN_HOST_ACC static auto mapIdx(
Vec<DimInt<TidxDimIn>, TElem> const&,
Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<0u>, TElem>
{
return {};
}
};
} 

ALPAKA_NO_HOST_ACC_WARNING template<
std::size_t TidxDimOut,
std::size_t TidxDimIn,
std::size_t TidxDimExtents,
typename TElem>
ALPAKA_FN_HOST_ACC auto mapIdx(
Vec<DimInt<TidxDimIn>, TElem> const& idx,
Vec<DimInt<TidxDimExtents>, TElem> const& extent) -> Vec<DimInt<TidxDimOut>, TElem>
{
return detail::MapIdx<TidxDimOut, TidxDimIn>::mapIdx(idx, extent);
}

namespace detail
{
template<std::size_t TidxDimOut, std::size_t TidxDimIn, typename TSfinae = void>
struct MapIdxPitchBytes;
template<std::size_t TidxDim>
struct MapIdxPitchBytes<TidxDim, TidxDim>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
Vec<DimInt<TidxDim>, TElem> const& idx,
[[maybe_unused]] Vec<DimInt<TidxDim>, TElem> const& pitch) -> Vec<DimInt<TidxDim>, TElem>
{
return idx;
}
};
template<std::size_t TidxDimOut>
struct MapIdxPitchBytes<TidxDimOut, 1u, std::enable_if_t<(TidxDimOut > 1u)>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
Vec<DimInt<1u>, TElem> const& idx,
Vec<DimInt<TidxDimOut>, TElem> const& pitch) -> Vec<DimInt<TidxDimOut>, TElem>
{
auto idxNd = Vec<DimInt<TidxDimOut>, TElem>::all(0u);

constexpr std::size_t lastIdx = TidxDimOut - 1u;

TElem tmp = idx[0u];
for(std::size_t d(0u); d < lastIdx; ++d)
{
idxNd[d] = static_cast<TElem>(tmp / pitch[d + 1]);
tmp %= pitch[d + 1];
}
idxNd[lastIdx] = tmp;

return idxNd;
}
};
template<std::size_t TidxDimIn>
struct MapIdxPitchBytes<1u, TidxDimIn, std::enable_if_t<(TidxDimIn > 1u)>>
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TElem>
ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
Vec<DimInt<TidxDimIn>, TElem> const& idx,
Vec<DimInt<TidxDimIn>, TElem> const& pitch) -> Vec<DimInt<1u>, TElem>
{
constexpr auto lastDim = TidxDimIn - 1;
TElem idx1d = idx[lastDim];
for(std::size_t d(0u); d < lastDim; ++d)
{
idx1d = static_cast<TElem>(idx1d + pitch[d + 1] * idx[d]);
}
return {idx1d};
}
};

template<std::size_t TidxDimOut>
struct MapIdxPitchBytes<TidxDimOut, 0u>
{
template<typename TElem, std::size_t TidxDimExtents>
ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
Vec<DimInt<0u>, TElem> const&,
Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<TidxDimOut>, TElem>
{
return Vec<DimInt<TidxDimOut>, TElem>::zeros();
}
};

template<std::size_t TidxDimIn>
struct MapIdxPitchBytes<0u, TidxDimIn>
{
template<typename TElem, std::size_t TidxDimExtents>
ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
Vec<DimInt<TidxDimIn>, TElem> const&,
Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<0u>, TElem>
{
return {};
}
};
} 

ALPAKA_NO_HOST_ACC_WARNING
template<std::size_t TidxDimOut, std::size_t TidxDimIn, std::size_t TidxDimPitch, typename TElem>
ALPAKA_FN_HOST_ACC auto mapIdxPitchBytes(
Vec<DimInt<TidxDimIn>, TElem> const& idx,
Vec<DimInt<TidxDimPitch>, TElem> const& pitch) -> Vec<DimInt<TidxDimOut>, TElem>
{
return detail::MapIdxPitchBytes<TidxDimOut, TidxDimIn>::mapIdxPitchBytes(idx, pitch);
}
} 
