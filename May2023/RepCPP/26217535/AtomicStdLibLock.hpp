

#pragma once

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <array>
#include <mutex>

namespace alpaka
{
template<size_t THashTableSize>
class AtomicStdLibLock
{
public:
template<typename TAtomic, typename TOp, typename T, typename THierarchy, typename TSfinae>
friend struct trait::AtomicOp;

static constexpr auto nextPowerOf2(size_t const value, size_t const bit = 0u) -> size_t
{
return value <= (static_cast<size_t>(1u) << bit) ? (static_cast<size_t>(1u) << bit)
: nextPowerOf2(value, bit + 1u);
}

template<typename TPtr>
static auto hash(TPtr const* const ptr) -> size_t
{
auto const ptrAddr = reinterpret_cast<size_t>(ptr);
constexpr size_t typeSizePowerOf2 = nextPowerOf2(sizeof(TPtr));
return (ptrAddr / typeSizePowerOf2);
}

template<typename TPtr>
auto getMutex(TPtr const* const ptr) const -> std::mutex&
{
constexpr size_t hashTableSize = THashTableSize == 0u ? 1u : nextPowerOf2(THashTableSize);

size_t const hashedAddr = hash(ptr) & (hashTableSize - 1u);
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
static std::array<
std::mutex,
hashTableSize>
m_mtxAtomic; 
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
return m_mtxAtomic[hashedAddr];
}
};

namespace trait
{
template<typename TOp, typename T, typename THierarchy, size_t THashTableSize>
struct AtomicOp<TOp, AtomicStdLibLock<THashTableSize>, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(
AtomicStdLibLock<THashTableSize> const& atomic,
T* const addr,
T const& value) -> T
{
std::lock_guard<std::mutex> lock(atomic.getMutex(addr));
return TOp()(addr, value);
}
ALPAKA_FN_HOST static auto atomicOp(
AtomicStdLibLock<THashTableSize> const& atomic,
T* const addr,
T const& compare,
T const& value) -> T
{
std::lock_guard<std::mutex> lock(atomic.getMutex(addr));
return TOp()(addr, compare, value);
}
};
} 
} 
