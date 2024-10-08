


#pragma once

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/util.h>

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {
namespace alignment_of_detail {


template <typename T>
class alignment_of_impl;

template <typename T, std::size_t size_diff>
struct helper
{
static const std::size_t value = size_diff;
};

template <typename T>
class helper<T, 0>
{
public:
static const std::size_t value = alignment_of_impl<T>::value;
};

template <typename T>
class alignment_of_impl
{
private:
struct big
{
T    x;
char c;
};

public:
static const std::size_t value = helper<big, sizeof(big) - sizeof(T)>::value;
};


}    


template <typename T>
struct alignment_of
: alignment_of_detail::alignment_of_impl<T>
{
};


template <std::size_t Align>
struct aligned_type;

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC


#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
template <>
struct aligned_type<1>
{
struct __align__(1) type{};
};

template <>
struct aligned_type<2>
{
struct __align__(2) type{};
};

template <>
struct aligned_type<4>
{
struct __align__(4) type{};
};

template <>
struct aligned_type<8>
{
struct __align__(8) type{};
};

template <>
struct aligned_type<16>
{
struct __align__(16) type{};
};

template <>
struct aligned_type<32>
{
struct __align__(32) type{};
};

template <>
struct aligned_type<64>
{
struct __align__(64) type{};
};

template <>
struct aligned_type<128>
{
struct __align__(128) type{};
};

template <>
struct aligned_type<256>
{
struct __align__(256) type{};
};

template <>
struct aligned_type<512>
{
struct __align__(512) type{};
};

template <>
struct aligned_type<1024>
{
struct __align__(1024) type{};
};

template <>
struct aligned_type<2048>
{
struct __align__(2048) type{};
};

template <>
struct aligned_type<4096>
{
struct __align__(4096) type{};
};

template <>
struct aligned_type<8192>
{
struct __align__(8192) type{};
};
#elif (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && (HYDRA_THRUST_GCC_VERSION < 40300)
template <>
struct aligned_type<1>
{
struct __align__(1) type{};
};

template <>
struct aligned_type<2>
{
struct __align__(2) type{};
};

template <>
struct aligned_type<4>
{
struct __align__(4) type{};
};

template <>
struct aligned_type<8>
{
struct __align__(8) type{};
};

template <>
struct aligned_type<16>
{
struct __align__(16) type{};
};

template <>
struct aligned_type<32>
{
struct __align__(32) type{};
};

template <>
struct aligned_type<64>
{
struct __align__(64) type{};
};

template <>
struct aligned_type<128>
{
struct __align__(128) type{};
};

#else
template <std::size_t Align>
struct aligned_type
{
struct __align__(Align) type{};
};
#endif    
#else
template <std::size_t Align>
struct aligned_type
{
struct type
{
};
};
#endif    


template <std::size_t Len, std::size_t Align>
struct aligned_storage
{
union type
{
unsigned char data[Len];

typename aligned_type<Align>::type align;
};
};


}    

HYDRA_THRUST_END_NS
