



#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <array>

namespace alpaka::trait
{
template<typename TElem, std::size_t Tsize>
struct DevType<std::array<TElem, Tsize>>
{
using type = DevCpu;
};

template<typename TElem, std::size_t Tsize>
struct GetDev<std::array<TElem, Tsize>>
{
ALPAKA_FN_HOST static auto getDev(std::array<TElem, Tsize> const& ) -> DevCpu
{
return getDevByIdx<PltfCpu>(0u);
}
};

template<typename TElem, std::size_t Tsize>
struct DimType<std::array<TElem, Tsize>>
{
using type = DimInt<1u>;
};

template<typename TElem, std::size_t Tsize>
struct ElemType<std::array<TElem, Tsize>>
{
using type = TElem;
};

template<typename TElem, std::size_t Tsize>
struct GetExtent<DimInt<0u>, std::array<TElem, Tsize>>
{
ALPAKA_FN_HOST static constexpr auto getExtent(std::array<TElem, Tsize> const& extent)
-> Idx<std::array<TElem, Tsize>>
{
return std::size(extent);
}
};

template<typename TElem, std::size_t Tsize>
struct GetPtrNative<std::array<TElem, Tsize>>
{
ALPAKA_FN_HOST static auto getPtrNative(std::array<TElem, Tsize> const& view) -> TElem const*
{
return std::data(view);
}
ALPAKA_FN_HOST static auto getPtrNative(std::array<TElem, Tsize>& view) -> TElem*
{
return std::data(view);
}
};

template<typename TIdx, typename TElem, std::size_t Tsize>
struct GetOffset<TIdx, std::array<TElem, Tsize>>
{
ALPAKA_FN_HOST static auto getOffset(std::array<TElem, Tsize> const&) -> Idx<std::array<TElem, Tsize>>
{
return 0u;
}
};

template<typename TElem, std::size_t Tsize>
struct IdxType<std::array<TElem, Tsize>>
{
using type = std::size_t;
};
} 
