

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

namespace alpaka::gb
{
template<typename TDim, typename TIdx>
class IdxGbGenericSycl : public concepts::Implements<ConceptIdxGb, IdxGbGenericSycl<TDim, TIdx>>
{
public:
using IdxGbBase = IdxGbGenericSycl;

explicit IdxGbGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item{work_item}
{
}

sycl::nd_item<TDim::value> my_item;
};
} 

namespace alpaka::trait
{
template<typename TDim, typename TIdx>
struct DimType<gb::IdxGbGenericSycl<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<gb::IdxGbGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
{
template<typename TWorkDiv>
static auto getIdx(gb::IdxGbGenericSycl<TDim, TIdx> const& idx, TWorkDiv const&)
{
if constexpr(TDim::value == 1)
return Vec<TDim, TIdx>(static_cast<TIdx>(idx.my_item.get_group(0)));
else if constexpr(TDim::value == 2)
{
return Vec<TDim, TIdx>(
static_cast<TIdx>(idx.my_item.get_group(1)),
static_cast<TIdx>(idx.my_item.get_group(0)));
}
else
{
return Vec<TDim, TIdx>(
static_cast<TIdx>(idx.my_item.get_group(2)),
static_cast<TIdx>(idx.my_item.get_group(1)),
static_cast<TIdx>(idx.my_item.get_group(0)));
}
}
};

template<typename TDim, typename TIdx>
struct IdxType<gb::IdxGbGenericSycl<TDim, TIdx>>
{
using type = TIdx;
};
} 

#endif
