#pragma once

#include <CL/sycl.hpp>
#include "cuda/Comparators.dp.hpp"
#include "cuda/CudaUtils.dp.hpp"
#include "cuda/ShuffleTypes.dp.hpp"

namespace facebook { namespace cuda {

namespace detail {

template <typename T, typename Comparator>
inline T shflSwap(const T x, int mask, int dir,
sycl::nd_item<1> &item) {
T y = shfl_xor(x, mask, item);
return Comparator::compare(x, y) == dir ? y : x;
}

} 

template <typename T, typename Comparator>
T warpBitonicSort(T val, sycl::nd_item<1> &item) {
const int laneId = getLaneId(item);
val = detail::shflSwap<T, Comparator>(
val, 0x01, getBit(laneId, 1) ^ getBit(laneId, 0), item);

val = detail::shflSwap<T, Comparator>(
val, 0x02, getBit(laneId, 2) ^ getBit(laneId, 1), item);
val = detail::shflSwap<T, Comparator>(
val, 0x01, getBit(laneId, 2) ^ getBit(laneId, 0), item);

val = detail::shflSwap<T, Comparator>(
val, 0x04, getBit(laneId, 3) ^ getBit(laneId, 2), item);
val = detail::shflSwap<T, Comparator>(
val, 0x02, getBit(laneId, 3) ^ getBit(laneId, 1), item);
val = detail::shflSwap<T, Comparator>(
val, 0x01, getBit(laneId, 3) ^ getBit(laneId, 0), item);

val = detail::shflSwap<T, Comparator>(
val, 0x08, getBit(laneId, 4) ^ getBit(laneId, 3), item);
val = detail::shflSwap<T, Comparator>(
val, 0x04, getBit(laneId, 4) ^ getBit(laneId, 2), item);
val = detail::shflSwap<T, Comparator>(
val, 0x02, getBit(laneId, 4) ^ getBit(laneId, 1), item);
val = detail::shflSwap<T, Comparator>(
val, 0x01, getBit(laneId, 4) ^ getBit(laneId, 0), item);

val = detail::shflSwap<T, Comparator>(val, 0x10, getBit(laneId, 4), item);
val = detail::shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 3), item);
val = detail::shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 2), item);
val = detail::shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 1), item);
val = detail::shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 0), item);

return val;
}

} } 
