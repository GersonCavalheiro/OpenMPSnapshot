


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/atomic/Traits.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>
#    include <type_traits>

namespace alpaka
{
class AtomicGenericSycl
{
};

namespace detail
{
template<typename THierarchy>
struct SyclMemoryScope
{
};

template<>
struct SyclMemoryScope<hierarchy::Grids>
{
static constexpr auto value = sycl::memory_scope::device;
};

template<>
struct SyclMemoryScope<hierarchy::Blocks>
{
static constexpr auto value = sycl::memory_scope::device;
};

template<>
struct SyclMemoryScope<hierarchy::Threads>
{
static constexpr auto value = sycl::memory_scope::work_group;
};

template<typename T>
inline auto get_global_ptr(T* const addr)
{
return sycl::make_ptr<T, sycl::access::address_space::global_space>(addr);
}

template<typename T>
inline auto get_local_ptr(T* const addr)
{
return sycl::make_ptr<T, sycl::access::address_space::local_space>(addr);
}

template<typename T, typename THierarchy>
using global_ref = sycl::atomic_ref<
T,
sycl::memory_order::relaxed,
SyclMemoryScope<THierarchy>::value,
sycl::access::address_space::global_space>;

template<typename T, typename THierarchy>
using local_ref = sycl::atomic_ref<
T,
sycl::memory_order::relaxed,
SyclMemoryScope<THierarchy>::value,
sycl::access::address_space::local_space>;

template<typename THierarchy, typename T, typename TOp>
inline auto callAtomicOp(T* const addr, TOp&& op)
{
if(auto ptr = get_global_ptr(addr); ptr != nullptr)
{
auto ref = global_ref<T, THierarchy>{*addr};
return op(ref);
}
else
{
auto ref = local_ref<T, THierarchy>{*addr};
return op(ref);
}
}

template<typename TRef, typename T, typename TEval>
inline auto casWithCondition(T* const addr, TEval&& eval)
{
auto ref = TRef{*addr};

auto old_val = ref.load();
auto assumed = T{};

do
{
assumed = old_val;
auto const new_val = eval(old_val);
old_val = ref.compare_exchange_strong(assumed, new_val);
} while(assumed != old_val);


return old_val;
}
} 
} 

namespace alpaka::trait
{
template<typename T, typename THierarchy>
struct AtomicOp<AtomicAdd, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_add(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicSub, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_sub(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMin, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_min(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMax, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_max(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicExch, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(addr, [&value](auto& ref) { return ref.exchange(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicInc, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_unsigned_v<T>, "atomicInc only supported for unsigned types");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
auto inc = [&value](auto old_val) { return (old_val >= value) ? static_cast<T>(0) : (old_val + 1u); };
if(auto ptr = alpaka::detail::get_global_ptr(addr); ptr != nullptr)
return alpaka::detail::casWithCondition<alpaka::detail::global_ref<T, THierarchy>>(addr, inc);
else
return alpaka::detail::casWithCondition<alpaka::detail::local_ref<T, THierarchy>>(addr, inc);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicDec, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_unsigned_v<T>, "atomicDec only supported for unsigned types");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
auto dec
= [&value](auto& old_val) { return ((old_val == 0) || (old_val > value)) ? value : (old_val - 1u); };
if(auto ptr = alpaka::detail::get_global_ptr(addr); ptr != nullptr)
return alpaka::detail::casWithCondition<alpaka::detail::global_ref<T, THierarchy>>(addr, dec);
else
return alpaka::detail::casWithCondition<alpaka::detail::local_ref<T, THierarchy>>(addr, dec);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicAnd, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_and(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicOr, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(addr, [&value](auto& ref) { return ref.fetch_or(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicXor, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
{
return alpaka::detail::callAtomicOp<THierarchy>(
addr,
[&value](auto& ref) { return ref.fetch_xor(value); });
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicCas, AtomicGenericSycl, T, THierarchy>
{
static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& compare, T const& value) -> T
{
auto cas = [&compare, &value](auto& ref)
{
auto tmp = compare;

const auto old = ref.load();

ref.compare_exchange_strong(tmp, value);

return old;
};

if(auto ptr = alpaka::detail::get_global_ptr(addr); ptr != nullptr)
{
auto ref = alpaka::detail::global_ref<T, THierarchy>{*addr};
return cas(ref);
}
else
{
auto ref = alpaka::detail::local_ref<T, THierarchy>{*addr};
return cas(ref);
}
}
};
} 

#endif
