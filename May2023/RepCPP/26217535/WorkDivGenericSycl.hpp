

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/Traits.hpp>

#    include <CL/sycl.hpp>

namespace alpaka
{
template<typename TDim, typename TIdx>
class WorkDivGenericSycl : public concepts::Implements<ConceptWorkDiv, WorkDivGenericSycl<TDim, TIdx>>
{
public:
using WorkDivBase = WorkDivGenericSycl;

WorkDivGenericSycl(Vec<TDim, TIdx> const& threadElemExtent, sycl::nd_item<TDim::value> work_item)
: m_threadElemExtent{threadElemExtent}
, my_item{work_item}
{
}

Vec<TDim, TIdx> const& m_threadElemExtent;
sycl::nd_item<TDim::value> my_item;
};
} 

namespace alpaka::trait
{
template<typename TDim, typename TIdx>
struct DimType<WorkDivGenericSycl<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct IdxType<WorkDivGenericSycl<TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
{
static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
{
if constexpr(TDim::value == 1)
return Vec<TDim, TIdx>{static_cast<TIdx>(workDiv.my_item.get_group_range(0))};
else if constexpr(TDim::value == 2)
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(workDiv.my_item.get_group_range(1)),
static_cast<TIdx>(workDiv.my_item.get_group_range(0))};
}
else
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(workDiv.my_item.get_group_range(2)),
static_cast<TIdx>(workDiv.my_item.get_group_range(1)),
static_cast<TIdx>(workDiv.my_item.get_group_range(0))};
}
}
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Block, unit::Threads>
{
static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
{
if constexpr(TDim::value == 1)
return Vec<TDim, TIdx>{static_cast<TIdx>(workDiv.my_item.get_local_range(0))};
else if constexpr(TDim::value == 2)
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(workDiv.my_item.get_local_range(1)),
static_cast<TIdx>(workDiv.my_item.get_local_range(0))};
}
else
{
return Vec<TDim, TIdx>{
static_cast<TIdx>(workDiv.my_item.get_local_range(2)),
static_cast<TIdx>(workDiv.my_item.get_local_range(1)),
static_cast<TIdx>(workDiv.my_item.get_local_range(0))};
}
}
};

template<typename TDim, typename TIdx>
struct GetWorkDiv<WorkDivGenericSycl<TDim, TIdx>, origin::Thread, unit::Elems>
{
static auto getWorkDiv(WorkDivGenericSycl<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
{
return workDiv.m_threadElemExtent;
}
};
} 

#endif
