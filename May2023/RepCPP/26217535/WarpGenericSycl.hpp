

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/warp/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka::warp
{
template<typename TDim>
class WarpGenericSycl : public concepts::Implements<alpaka::warp::ConceptWarp, WarpGenericSycl<TDim>>
{
public:
WarpGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item{my_item}
{
}

sycl::nd_item<TDim::value> m_item;
};
} 

namespace alpaka::warp::trait
{
template<typename TDim>
struct GetSize<warp::WarpGenericSycl<TDim>>
{
static auto getSize(warp::WarpGenericSycl<TDim> const& warp) -> std::int32_t
{
auto const sub_group = warp.m_item.get_sub_group();
return static_cast<std::int32_t>(sub_group.get_local_linear_range());
}
};

template<typename TDim>
struct Activemask<warp::WarpGenericSycl<TDim>>
{
static auto activemask(warp::WarpGenericSycl<TDim> const& warp) -> std::uint32_t
{
auto const sub_group = warp.m_item.get_sub_group();
return sycl::ext::oneapi::group_ballot(sub_group, true);
}
};

template<typename TDim>
struct All<warp::WarpGenericSycl<TDim>>
{
static auto all(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
{
auto const sub_group = warp.m_item.get_sub_group();
return static_cast<std::int32_t>(sycl::all_of_group(sub_group, static_cast<bool>(predicate)));
}
};

template<typename TDim>
struct Any<warp::WarpGenericSycl<TDim>>
{
static auto any(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
{
auto const sub_group = warp.m_item.get_sub_group();
return static_cast<std::int32_t>(sycl::any_of_group(sub_group, static_cast<bool>(predicate)));
}
};

template<typename TDim>
struct Ballot<warp::WarpGenericSycl<TDim>>
{
static auto ballot(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate)
{
auto const sub_group = warp.m_item.get_sub_group();
return sycl::ext::oneapi::group_ballot(sub_group, static_cast<bool>(predicate));
}
};

template<typename TDim>
struct Shfl<warp::WarpGenericSycl<TDim>>
{
template<typename T>
static auto shfl(warp::WarpGenericSycl<TDim> const& warp, T value, std::int32_t srcLane, std::int32_t width)
{

auto const actual_group = warp.m_item.get_sub_group();
auto const actual_item_id = actual_group.get_local_linear_id();

auto const assumed_group_id = actual_item_id / width;
auto const assumed_item_id = actual_item_id % width;

auto const assumed_src_id = static_cast<std::size_t>(srcLane % width);
auto const actual_src_id = assumed_src_id + assumed_group_id * width;

auto const src = sycl::id<1>{actual_src_id};

return sycl::select_from_group(actual_group, value, src);
}
};
} 

#endif
