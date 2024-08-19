#pragma once

#include <CL/sycl.hpp>
#include "cuda/CudaUtils.dp.hpp"
#include "cuda/DeviceTensor.dp.hpp"
#include "cuda/Pair.dp.hpp"
#include "cuda/ShuffleTypes.dp.hpp"

#include <boost/preprocessor/repetition/repeat.hpp>

namespace facebook { namespace cuda {

template <typename T, int N>
struct RegisterUtils {
template <int Shift> inline static void shiftLeft(T arr[N]) {
#pragma unroll
for (int i = 0; i < N - Shift; ++i) {
arr[i] = arr[i + Shift];
}
}

template <int Shift> inline static void shiftRight(T arr[N]) {
#pragma unroll
for (int i = N - 1; i >= Shift; --i) {
arr[i] = arr[i - Shift];
}
}

template <int Rotate> inline static void rotateLeft(T arr[N]) {
T tmp[Rotate];

#pragma unroll
for (int i = 0; i < Rotate; ++i) {
tmp[i] = arr[i];
}

#pragma unroll
for (int i = 0; i < N - Rotate; ++i) {
arr[i] = arr[i + Rotate];
}

#pragma unroll
for (int i = 0; i < Rotate; ++i) {
arr[N - Rotate + i] = tmp[i];
}
}

template <int Rotate> inline static void rotateRight(T arr[N]) {
T tmp[Rotate];

#pragma unroll
for (int i = 0; i < Rotate; ++i) {
tmp[i] = arr[N - Rotate + i];
}

#pragma unroll
for (int i = N - 1; i >= Rotate; --i) {
arr[i] = arr[i - Rotate];
}

#pragma unroll
for (int i = 0; i < Rotate; ++i) {
arr[i] = tmp[i];
}
}
};


template <typename T, int N>
struct RegisterIndexUtils {
inline static T get(const T arr[N], int index);

inline static void set(T arr[N], int index, T val);
};

template <typename T, int N>
struct WarpRegisterUtils {
static T broadcast(const T arr[N], int index, sycl::nd_item<1> &item) {
const int lane = index & (WARP_SIZE - 1);
const int bucket = index / WARP_SIZE;

return shfl(RegisterIndexUtils<T, N>::get(arr, bucket), lane, item);
}
};

template <typename T, int N>
struct WarpRegisterLoaderUtils {
static void load(T arr[N],
const DeviceTensor<T, 1>& in,
const T fill,
sycl::nd_item<1> &item) {
const int lane = getLaneId(item);

for (int i = 0; i < N; ++i) {
const int offset = lane + i * WARP_SIZE;
arr[i] = (offset < in.getSize(0)) ? in[offset] : fill;
}
}

static void save(DeviceTensor<T, 1>& out,
const T arr[N],
const int num,
sycl::nd_item<1> &item) {
const int lane = getLaneId(item);

for (int i = 0; i < N; ++i) {
const int offset = lane + i * WARP_SIZE;
if (offset < num) {
out[offset] = arr[i];
}
}
}
};

template <typename K, typename V, int N>
struct WarpRegisterPairLoaderUtils {
static void load(Pair<K, V> arr[N],
const DeviceTensor<K, 1>& in,
const K keyFill,
const V valueFill,
sycl::nd_item<1> &item) {
const int lane = getLaneId(item);

for (int i = 0; i < N; ++i) {
const int offset = lane + i * WARP_SIZE;
arr[i] = (offset < in.getSize(0)) ?
Pair<K, V>(in[offset], offset) : Pair<K, V>(keyFill, valueFill);
}
}

static void load(Pair<K, V> arr[N],
const DeviceTensor<K, 1>& key,
const DeviceTensor<V, 1>& value,
const K keyFill,
const V valueFill,
sycl::nd_item<1> &item) {
const int lane = getLaneId(item);

for (int i = 0; i < N; ++i) {
const int offset = lane + i * WARP_SIZE;
arr[i] = (offset < key.getSize(0)) ?
Pair<K, V>(key[offset], value[offset]) : Pair<K, V>(keyFill, valueFill);
}
}

static void save(DeviceTensor<K, 1>& key,
DeviceTensor<V, 1>& value,
const Pair<K, V> arr[N],
const int num,
sycl::nd_item<1> &item) {
const int lane = getLaneId(item);

for (int i = 0; i < N; ++i) {
const int offset = lane + i * WARP_SIZE;

if (offset < num) {
key[offset] = arr[i].k;
value[offset] = arr[i].v;
}
}
}
};

#define GET_CASE(UNUSED1, I, UNUSED2)           \
case I:                                       \
return arr[I];

#define SET_CASE(UNUSED1, I, UNUSED2)           \
case I:                                       \
arr[I] = val;                                 \
break;

#define IMPL_REGISTER_ARRAY(N)                                                 \
template <typename T> struct RegisterIndexUtils<T, N> {                      \
inline static T get(const T arr[N], int index) {                           \
switch (index) {                                                         \
BOOST_PP_REPEAT(N, GET_CASE, 0);                                       \
default:                                                                 \
return T();                                                            \
};                                                                       \
}                                                                          \
\
inline static void set(T arr[N], int index, T val) {                       \
switch (index) { BOOST_PP_REPEAT(N, SET_CASE, 0); }                      \
}                                                                          \
};

#define IMPL_REGISTER_ARRAY_CASE(UNUSED1, I, UNUSED2) IMPL_REGISTER_ARRAY(I);

BOOST_PP_REPEAT(32, IMPL_REGISTER_ARRAY_CASE, 0);

} } 
