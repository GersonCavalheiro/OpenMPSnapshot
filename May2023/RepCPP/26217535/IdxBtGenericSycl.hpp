

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

namespace alpaka::bt
{
template<typename TDim, typename TIdx>
class IdxBtGenericSycl : public concepts::Implements<ConceptIdxBt, IdxBtGenericSycl<TDim, TIdx>>
{
public:
using IdxBtBase = IdxBtGenericSycl;

explicit IdxBtGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item{work_item}
{
}

sycl::nd_item<TDim::value> my_item;
};
} 

namespace alpaka::trait
{
template<typename TDim, typename TIdx>
struct DimType<bt::IdxBtGenericSycl<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtGenericSycl<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
static auto getIdx(bt::IdxBtGenericSycl<TDim, TIdx> const& idx, TWorkDiv const&) -> Vec<TDim, TIdx>
{
if constexpr(TDim::value == 1)
return Vec<TDim, TIdx>{static_cast<TIdx>(idx.my_item.get_local_id(0))};
else if constexpr(TDim::value == 2)
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(idx.my_item.get_local_id(1)),
static_cast<TIdx>(idx.my_item.get_local_id(0))};
}
else
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(idx.my_item.get_local_id(2)),
static_cast<TIdx>(idx.my_item.get_local_id(1)),
static_cast<TIdx>(idx.my_item.get_local_id(0))};
}
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtGenericSycl<TDim, TIdx>>
{
using type = TIdx;
};
} 

#endif
