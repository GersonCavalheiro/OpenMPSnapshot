

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/intrinsic/IntrinsicFallback.hpp>
#    include <alpaka/intrinsic/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka
{
class IntrinsicGenericSycl : public concepts::Implements<ConceptIntrinsic, IntrinsicGenericSycl>
{
};
} 

namespace alpaka::trait
{
template<>
struct Popcount<IntrinsicGenericSycl>
{
static auto popcount(IntrinsicGenericSycl const&, std::uint32_t value) -> std::int32_t
{
return static_cast<std::int32_t>(sycl::popcount(value));
}

static auto popcount(IntrinsicGenericSycl const&, std::uint64_t value) -> std::int32_t
{
return static_cast<std::int32_t>(sycl::popcount(value));
}
};

template<>
struct Ffs<IntrinsicGenericSycl>
{
static auto ffs(IntrinsicGenericSycl const&, std::int32_t value) -> std::int32_t
{
return (value == 0) ? 0 : sycl::popcount(value ^ ~(-value));
}

static auto ffs(IntrinsicGenericSycl const&, std::int64_t value) -> std::int32_t
{
return (value == 0l) ? 0 : static_cast<std::int32_t>(sycl::popcount(value ^ ~(-value)));
}
};
} 

#endif
