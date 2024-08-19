



#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <vector>

namespace alpaka::trait
{
template<typename TElem, typename TAllocator>
struct DevType<std::vector<TElem, TAllocator>>
{
using type = DevCpu;
};

template<typename TElem, typename TAllocator>
struct GetDev<std::vector<TElem, TAllocator>>
{
ALPAKA_FN_HOST static auto getDev(std::vector<TElem, TAllocator> const& ) -> DevCpu
{
return getDevByIdx<PltfCpu>(0u);
}
};

template<typename TElem, typename TAllocator>
struct DimType<std::vector<TElem, TAllocator>>
{
using type = DimInt<1u>;
};

template<typename TElem, typename TAllocator>
struct ElemType<std::vector<TElem, TAllocator>>
{
using type = TElem;
};

template<typename TElem, typename TAllocator>
struct GetExtent<DimInt<0u>, std::vector<TElem, TAllocator>>
{
ALPAKA_FN_HOST static auto getExtent(std::vector<TElem, TAllocator> const& extent)
-> Idx<std::vector<TElem, TAllocator>>
{
return std::size(extent);
}
};

template<typename TElem, typename TAllocator>
struct GetPtrNative<std::vector<TElem, TAllocator>>
{
ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator> const& view) -> TElem const*
{
return std::data(view);
}
ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator>& view) -> TElem*
{
return std::data(view);
}
};

template<typename TIdx, typename TElem, typename TAllocator>
struct GetOffset<TIdx, std::vector<TElem, TAllocator>>
{
ALPAKA_FN_HOST static auto getOffset(std::vector<TElem, TAllocator> const&)
-> Idx<std::vector<TElem, TAllocator>>
{
return 0u;
}
};

template<typename TElem, typename TAllocator>
struct IdxType<std::vector<TElem, TAllocator>>
{
using type = std::size_t;
};
} 
