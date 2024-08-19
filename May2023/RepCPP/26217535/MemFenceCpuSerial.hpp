

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/mem/fence/Traits.hpp>

#include <atomic>

namespace alpaka
{
class MemFenceCpuSerial : public concepts::Implements<ConceptMemFence, MemFenceCpuSerial>
{
};

namespace trait
{
template<>
struct MemFence<MemFenceCpuSerial, memory_scope::Block>
{
static auto mem_fence(MemFenceCpuSerial const&, memory_scope::Block const&)
{

}
};

template<>
struct MemFence<MemFenceCpuSerial, memory_scope::Grid>
{
static auto mem_fence(MemFenceCpuSerial const&, memory_scope::Grid const&)
{

}
};

template<typename TMemScope>
struct MemFence<MemFenceCpuSerial, TMemScope>
{
static auto mem_fence(MemFenceCpuSerial const&, TMemScope const&)
{


static std::atomic<int> dummy{42};


auto x = dummy.load(std::memory_order_relaxed);
std::atomic_thread_fence(std::memory_order_acq_rel);
dummy.store(x, std::memory_order_relaxed);
}
};
} 
} 
