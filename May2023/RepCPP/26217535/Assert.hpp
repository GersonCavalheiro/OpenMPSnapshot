

#pragma once

#include <alpaka/core/Common.hpp>

#include <cassert>
#include <type_traits>

#define ALPAKA_ASSERT(...) assert(__VA_ARGS__)

#if defined(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST) || defined(SYCL_EXT_ONEAPI_ASSERT)
#    define ALPAKA_ASSERT_OFFLOAD(EXPRESSION) ALPAKA_ASSERT(EXPRESSION)
#elif defined __AMDGCN__ && (!defined NDEBUG)
#    define ALPAKA_ASSERT_OFFLOAD(EXPRESSION)                                                                         \
do                                                                                                            \
{                                                                                                             \
if(!(EXPRESSION))                                                                                         \
__builtin_trap();                                                                                     \
} while(false)
#else
#    define ALPAKA_ASSERT_OFFLOAD(EXPRESSION)                                                                         \
do                                                                                                            \
{                                                                                                             \
} while(false)
#endif

namespace alpaka::core
{
namespace detail
{
template<typename TArg>
struct AssertValueUnsigned
{
ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static constexpr auto assertValueUnsigned(
[[maybe_unused]] TArg const& arg)
{
if constexpr(std::is_signed_v<TArg>)
ALPAKA_ASSERT_OFFLOAD(arg >= 0);

}
};
} 

ALPAKA_NO_HOST_ACC_WARNING
template<typename TArg>
ALPAKA_FN_HOST_ACC constexpr auto assertValueUnsigned(TArg const& arg) -> void
{
detail::AssertValueUnsigned<TArg>::assertValueUnsigned(arg);
}

namespace detail
{
template<typename TLhs, typename TRhs>
struct AssertGreaterThan
{
ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static constexpr auto assertGreaterThan(
[[maybe_unused]] TRhs const& rhs)
{
if constexpr(std::is_signed_v<TRhs> || (TLhs::value != 0u))
ALPAKA_ASSERT_OFFLOAD(TLhs::value > rhs);

}
};
} 

ALPAKA_NO_HOST_ACC_WARNING
template<typename TLhs, typename TRhs>
ALPAKA_FN_HOST_ACC constexpr auto assertGreaterThan(TRhs const& rhs) -> void
{
detail::AssertGreaterThan<TLhs, TRhs>::assertGreaterThan(rhs);
}
} 
