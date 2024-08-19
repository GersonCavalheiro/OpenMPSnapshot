

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/mem/fence/Traits.hpp>

#include <atomic>

namespace alpaka
{
class MemFenceCpu : public concepts::Implements<ConceptMemFence, MemFenceCpu>
{
};

namespace trait
{
template<typename TMemScope>
struct MemFence<MemFenceCpu, TMemScope>
{
static auto mem_fence(MemFenceCpu const&, TMemScope const&)
{


static auto dummy = std::atomic<int>{42};


auto x = dummy.load(std::memory_order_relaxed);
std::atomic_thread_fence(std::memory_order_acq_rel);
dummy.store(x, std::memory_order_relaxed);
}
};
} 
} 
