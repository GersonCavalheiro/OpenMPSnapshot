

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/elem/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/meta/IntegerSequence.hpp>
#    include <alpaka/offset/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <array>
#    include <cstddef>
#    include <iostream>
#    include <stdexcept>
#    include <string>
#    include <type_traits>
#    include <utility>

namespace alpaka
{
namespace detail
{
template<typename T, typename... Ts>
struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)>
{
};
} 

template<typename T>
struct IsSyclBuiltInType
: detail::is_any<
T,
sycl::half,

sycl::char2,
sycl::schar2,
sycl::uchar2,
sycl::short2,
sycl::ushort2,
sycl::int2,
sycl::uint2,
sycl::long2,
sycl::ulong2,
sycl::longlong2,
sycl::ulonglong2,
sycl::float2,
sycl::double2,
sycl::half2,

sycl::char3,
sycl::schar3,
sycl::uchar3,
sycl::short3,
sycl::ushort3,
sycl::int3,
sycl::uint3,
sycl::long3,
sycl::ulong3,
sycl::longlong3,
sycl::ulonglong3,
sycl::float3,
sycl::double3,
sycl::half3,

sycl::char4,
sycl::schar4,
sycl::uchar4,
sycl::short4,
sycl::ushort4,
sycl::int4,
sycl::uint4,
sycl::long4,
sycl::ulong4,
sycl::longlong4,
sycl::ulonglong4,
sycl::float4,
sycl::double4,
sycl::half4,

sycl::char8,
sycl::schar8,
sycl::uchar8,
sycl::short8,
sycl::ushort8,
sycl::int8,
sycl::uint8,
sycl::long8,
sycl::ulong8,
sycl::longlong8,
sycl::ulonglong8,
sycl::float8,
sycl::double8,
sycl::half8,

sycl::char16,
sycl::schar16,
sycl::uchar16,
sycl::short16,
sycl::ushort16,
sycl::int16,
sycl::uint16,
sycl::long16,
sycl::ulong16,
sycl::longlong16,
sycl::ulonglong16,
sycl::float16,
sycl::double16,
sycl::half16>
{
};
} 

namespace alpaka::trait
{
template<typename T>
struct DimType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
{
using type = std::conditional_t<std::is_scalar_v<T>, DimInt<std::size_t{1}>, DimInt<T::size()>>;
};

template<typename T>
struct ElemType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
{
using type = std::conditional_t<std::is_scalar_v<T>, T, typename T::element_type>;
};
} 

namespace alpaka::trait
{
template<typename TExtent>
struct GetExtent<DimInt<Dim<TExtent>::value>, TExtent, std::enable_if_t<IsSyclBuiltInType<TExtent>::value>>
{
static auto getExtent(TExtent const& extent)
{
if constexpr(std::is_scalar_v<TExtent>)
return extent;
else
{
return extent.template swizzle<DimInt<Dim<TExtent>::value>::value>();
}
}
};

template<typename TExtent, typename TExtentVal>
struct SetExtent<
DimInt<Dim<TExtent>::value>,
TExtent,
TExtentVal,
std::enable_if_t<IsSyclBuiltInType<TExtent>::value>>
{
static auto setExtent(TExtent const& extent, TExtentVal const& extentVal)
{
if constexpr(std::is_scalar_v<TExtent>)
extent = extentVal;
else
{
extent.template swizzle<DimInt<Dim<TExtent>::value>::value>() = extentVal;
}
}
};

template<typename TOffsets>
struct GetOffset<DimInt<Dim<TOffsets>::value>, TOffsets, std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
{
static auto getOffset(TOffsets const& offsets)
{
if constexpr(std::is_scalar_v<TOffsets>)
return offsets;
else
{
return offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>();
}
}
};

template<typename TOffsets, typename TOffset>
struct SetOffset<
DimInt<Dim<TOffsets>::value>,
TOffsets,
TOffset,
std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
{
static auto setOffset(TOffsets const& offsets, TOffset const& offset)
{
if constexpr(std::is_scalar_v<TOffsets>)
offsets = offset;
else
{
offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>() = offset;
}
}
};

template<typename TIdx>
struct IdxType<TIdx, std::enable_if_t<IsSyclBuiltInType<TIdx>::value>>
{
using type = std::size_t;
};
} 

#endif
