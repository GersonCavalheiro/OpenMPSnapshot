


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/atomic/AtomicGenericSycl.hpp>
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStGenericSycl.hpp>
#    include <alpaka/block/sync/BlockSyncGenericSycl.hpp>
#    include <alpaka/idx/bt/IdxBtGenericSycl.hpp>
#    include <alpaka/idx/gb/IdxGbGenericSycl.hpp>
#    include <alpaka/intrinsic/IntrinsicGenericSycl.hpp>
#    include <alpaka/math/MathGenericSycl.hpp>
#    include <alpaka/mem/fence/MemFenceGenericSycl.hpp>
#    include <alpaka/warp/WarpGenericSycl.hpp>
#    include <alpaka/workdiv/WorkDivGenericSycl.hpp>

#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Sycl.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>
#    include <string>
#    include <type_traits>

namespace alpaka
{
template<typename TDim, typename TIdx>
class AccGenericSycl
: public WorkDivGenericSycl<TDim, TIdx>
, public gb::IdxGbGenericSycl<TDim, TIdx>
, public bt::IdxBtGenericSycl<TDim, TIdx>
, public AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>
, public math::MathGenericSycl
, public BlockSharedMemDynGenericSycl
, public BlockSharedMemStGenericSycl
, public BlockSyncGenericSycl<TDim>
, public IntrinsicGenericSycl
, public MemFenceGenericSycl
, public warp::WarpGenericSycl<TDim>
{
public:
AccGenericSycl(AccGenericSycl const&) = delete;
AccGenericSycl(AccGenericSycl&&) = delete;
auto operator=(AccGenericSycl const&) -> AccGenericSycl& = delete;
auto operator=(AccGenericSycl&&) -> AccGenericSycl& = delete;

#    ifdef ALPAKA_SYCL_IOSTREAM_ENABLED
AccGenericSycl(
Vec<TDim, TIdx> const& threadElemExtent,
sycl::nd_item<TDim::value> work_item,
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> dyn_shared_acc,
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> st_shared_acc,
sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::global_buffer> global_fence_dummy,
sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> local_fence_dummy,
sycl::stream output_stream)
: WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item}
, gb::IdxGbGenericSycl<TDim, TIdx>{work_item}
, bt::IdxBtGenericSycl<TDim, TIdx>{work_item}
, AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{}
, math::MathGenericSycl{}
, BlockSharedMemDynGenericSycl{dyn_shared_acc}
, BlockSharedMemStGenericSycl{st_shared_acc}
, BlockSyncGenericSycl<TDim>{work_item}
, IntrinsicGenericSycl{}
, MemFenceGenericSycl{global_fence_dummy, local_fence_dummy}
, warp::WarpGenericSycl<TDim>{work_item}
, cout{output_stream}
{
}

sycl::stream cout;
#    else
AccGenericSycl(
Vec<TDim, TIdx> const& threadElemExtent,
sycl::nd_item<TDim::value> work_item,
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> dyn_shared_acc,
sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> st_shared_acc,
sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::global_buffer> global_fence_dummy,
sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> local_fence_dummy)
: WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item}
, gb::IdxGbGenericSycl<TDim, TIdx>{work_item}
, bt::IdxBtGenericSycl<TDim, TIdx>{work_item}
, AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{}
, math::MathGenericSycl{}
, BlockSharedMemDynGenericSycl{dyn_shared_acc}
, BlockSharedMemStGenericSycl{st_shared_acc}
, BlockSyncGenericSycl<TDim>{work_item}
, IntrinsicGenericSycl{}
, MemFenceGenericSycl{global_fence_dummy, local_fence_dummy}
, warp::WarpGenericSycl<TDim>{work_item}
{
}
#    endif
};
} 

namespace alpaka::trait
{
template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
struct AccType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
{
using type = TAcc<TDim, TIdx>;
};

template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
struct GetAccDevProps<
TAcc<TDim, TIdx>,
std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
{
static auto getAccDevProps(typename DevType<TAcc<TDim, TIdx>>::type const& dev) -> AccDevProps<TDim, TIdx>
{
auto const device = dev.getNativeHandle().first;
auto max_threads_dim = device.template get_info<sycl::info::device::max_work_item_sizes>();
return {
alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_compute_units>()),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
std::numeric_limits<TIdx>::max(),
std::numeric_limits<TIdx>::max(),
std::numeric_limits<TIdx>::max())),
std::numeric_limits<TIdx>::max(),
getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
alpaka::core::clipCast<TIdx>(max_threads_dim[2u]),
alpaka::core::clipCast<TIdx>(max_threads_dim[1u]),
alpaka::core::clipCast<TIdx>(max_threads_dim[0u]))),
alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_work_group_size>()),
Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
std::numeric_limits<TIdx>::max(),
device.template get_info<sycl::info::device::local_mem_size>()};
}
};

template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
struct DimType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
{
using type = TDim;
};

template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
struct IdxType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
{
using type = TIdx;
};
} 

#endif
