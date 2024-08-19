

#ifndef GKO_HIP_COMPONENTS_COOPERATIVE_GROUPS_HIP_HPP_
#define GKO_HIP_COMPONENTS_COOPERATIVE_GROUPS_HIP_HPP_


#include <type_traits>


#include "hip/base/config.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {



namespace group {




namespace detail {


template <typename T>
struct is_group_impl : std::false_type {};


template <typename T>
struct is_synchronizable_group_impl : std::false_type {};


template <typename T>
struct is_communicator_group_impl : std::true_type {};

}  



template <typename T>
using is_group = detail::is_group_impl<std::decay_t<T>>;



template <typename T>
using is_synchronizable_group =
detail::is_synchronizable_group_impl<std::decay_t<T>>;



template <typename T>
using is_communicator_group =
detail::is_communicator_group_impl<std::decay_t<T>>;




namespace detail {



template <unsigned Size>
class thread_block_tile {

static constexpr auto lane_mask_base = ~config::lane_mask_type{} >>
(config::warp_size - Size);

public:
__device__ thread_block_tile() : data_{Size, 0, 0, lane_mask_base}
{
auto tid =
unsigned(threadIdx.x +
blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
data_.rank = tid % Size;
data_.lane_offset = (tid % config::warp_size) / Size * Size;
data_.mask <<= data_.lane_offset;
}

__device__ __forceinline__ unsigned thread_rank() const noexcept
{
return data_.rank;
}

__device__ __forceinline__ unsigned size() const noexcept { return Size; }

__device__ __forceinline__ void sync() const noexcept
{
#if GINKGO_HIP_PLATFORM_NVCC
__syncwarp(data_.mask);
#endif  
}

#if GINKGO_HIP_PLATFORM_HCC
#define GKO_BIND_SHFL(ShflOp, ValueType, SelectorType)                       \
__device__ __forceinline__ ValueType ShflOp(                             \
ValueType var, SelectorType selector) const noexcept                 \
{                                                                        \
return __##ShflOp(var, selector, Size);                              \
}                                                                        \
static_assert(true,                                                      \
"This assert is used to counter the false positive extra " \
"semi-colon warnings")
#else
#define GKO_BIND_SHFL(ShflOp, ValueType, SelectorType)                       \
__device__ __forceinline__ ValueType ShflOp(                             \
ValueType var, SelectorType selector) const noexcept                 \
{                                                                        \
return __##ShflOp##_sync(data_.mask, var, selector, Size);           \
}                                                                        \
static_assert(true,                                                      \
"This assert is used to counter the false positive extra " \
"semi-colon warnings")
#endif

GKO_BIND_SHFL(shfl, int32, int32);
GKO_BIND_SHFL(shfl, float, int32);
GKO_BIND_SHFL(shfl, uint32, int32);
GKO_BIND_SHFL(shfl, double, int32);

GKO_BIND_SHFL(shfl_up, int32, uint32);
GKO_BIND_SHFL(shfl_up, uint32, uint32);
GKO_BIND_SHFL(shfl_up, float, uint32);
GKO_BIND_SHFL(shfl_up, double, uint32);

GKO_BIND_SHFL(shfl_down, int32, uint32);
GKO_BIND_SHFL(shfl_down, uint32, uint32);
GKO_BIND_SHFL(shfl_down, float, uint32);
GKO_BIND_SHFL(shfl_down, double, uint32);

GKO_BIND_SHFL(shfl_xor, int32, int32);
GKO_BIND_SHFL(shfl_xor, float, int32);
GKO_BIND_SHFL(shfl_xor, uint32, int32);
GKO_BIND_SHFL(shfl_xor, double, int32);


__device__ __forceinline__ int any(int predicate) const noexcept
{
#if GINKGO_HIP_PLATFORM_HCC
if (Size == config::warp_size) {
return __any(predicate);
} else {
return (__ballot(predicate) & data_.mask) != 0;
}
#else
return __any_sync(data_.mask, predicate);
#endif
}


__device__ __forceinline__ int all(int predicate) const noexcept
{
#if GINKGO_HIP_PLATFORM_HCC
if (Size == config::warp_size) {
return __all(predicate);
} else {
return (__ballot(predicate) & data_.mask) == data_.mask;
}
#else
return __all_sync(data_.mask, predicate);
#endif
}


__device__ __forceinline__ config::lane_mask_type ballot(
int predicate) const noexcept
{
#if GINKGO_HIP_PLATFORM_HCC
if (Size == config::warp_size) {
return __ballot(predicate);
} else {
return (__ballot(predicate) & data_.mask) >> data_.lane_offset;
}
#else
if (Size == config::warp_size) {
return __ballot_sync(data_.mask, predicate);
} else {
return __ballot_sync(data_.mask, predicate) >> data_.lane_offset;
}
#endif
}

private:
struct alignas(8) {
unsigned size;
unsigned rank;
unsigned lane_offset;
config::lane_mask_type mask;
} data_;
};


}  


namespace detail {


template <typename Group>
class enable_extended_shuffle : public Group {
public:
using Group::Group;
using Group::shfl;
using Group::shfl_down;
using Group::shfl_up;
using Group::shfl_xor;

#define GKO_ENABLE_SHUFFLE_OPERATION(_name, SelectorType)                   \
template <typename ValueType>                                           \
__device__ __forceinline__ ValueType _name(const ValueType& var,        \
SelectorType selector) const \
{                                                                       \
return shuffle_impl(                                                \
[this](uint32 v, SelectorType s) {                              \
return static_cast<const Group*>(this)->_name(v, s);        \
},                                                              \
var, selector);                                                 \
}

GKO_ENABLE_SHUFFLE_OPERATION(shfl, int32)
GKO_ENABLE_SHUFFLE_OPERATION(shfl_up, uint32)
GKO_ENABLE_SHUFFLE_OPERATION(shfl_down, uint32)
GKO_ENABLE_SHUFFLE_OPERATION(shfl_xor, int32)

#undef GKO_ENABLE_SHUFFLE_OPERATION

private:
template <typename ShuffleOperator, typename ValueType,
typename SelectorType>
static __device__ __forceinline__ ValueType
shuffle_impl(ShuffleOperator intrinsic_shuffle, const ValueType var,
SelectorType selector)
{
static_assert(sizeof(ValueType) % sizeof(uint32) == 0,
"Unable to shuffle sizes which are not 4-byte multiples");
constexpr auto value_size = sizeof(ValueType) / sizeof(uint32);
ValueType result;
auto var_array = reinterpret_cast<const uint32*>(&var);
auto result_array = reinterpret_cast<uint32*>(&result);
#pragma unroll
for (std::size_t i = 0; i < value_size; ++i) {
result_array[i] = intrinsic_shuffle(var_array[i], selector);
}
return result;
}
};


}  


template <unsigned Size>
struct thread_block_tile
: detail::enable_extended_shuffle<detail::thread_block_tile<Size>> {
using detail::enable_extended_shuffle<
detail::thread_block_tile<Size>>::enable_extended_shuffle;
};


template <size_type Size, typename Group>
__device__ __forceinline__
std::enable_if_t<(Size <= kernels::hip::config::warp_size) && (Size > 0) &&
(kernels::hip::config::warp_size % Size == 0),
thread_block_tile<Size>>
tiled_partition(const Group&)
{
return thread_block_tile<Size>();
}


namespace detail {


template <unsigned Size>
struct is_group_impl<thread_block_tile<Size>> : std::true_type {};
template <unsigned Size>
struct is_synchronizable_group_impl<thread_block_tile<Size>> : std::true_type {
};
template <unsigned Size>
struct is_communicator_group_impl<thread_block_tile<Size>> : std::true_type {};


}  


class thread_block {
friend __device__ __forceinline__ thread_block this_thread_block();

public:
__device__ __forceinline__ unsigned thread_rank() const noexcept
{
return data_.rank;
}

__device__ __forceinline__ unsigned size() const noexcept
{
return data_.size;
}

__device__ __forceinline__ void sync() const noexcept { __syncthreads(); }

private:
__device__ thread_block()
: data_{static_cast<unsigned>(blockDim.x * blockDim.y * blockDim.z),
static_cast<unsigned>(
threadIdx.x +
blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))}
{}
struct alignas(8) {
unsigned size;
unsigned rank;
} data_;
};


__device__ __forceinline__ thread_block this_thread_block()
{
return thread_block();
}


namespace detail {

template <>
struct is_group_impl<thread_block> : std::true_type {};
template <>
struct is_synchronizable_group_impl<thread_block> : std::true_type {};


}  



class grid_group {
friend __device__ grid_group this_grid();

public:
__device__ unsigned size() const noexcept { return data_.size; }

__device__ unsigned thread_rank() const noexcept { return data_.rank; }

private:
__device__ grid_group()
: data_{
blockDim.x * blockDim.y * blockDim.z *
gridDim.x * gridDim.y * gridDim.z,
threadIdx.x + blockDim.x *
(threadIdx.y + blockDim.y *
(threadIdx.z + blockDim.z *
(blockIdx.x + gridDim.x *
(blockIdx.y + gridDim.y * blockIdx.z))))}
{}

struct alignas(8) {
unsigned size;
unsigned rank;
} data_;
};

__device__ inline grid_group this_grid() { return {}; }


}  
}  
}  
}  


#endif  
