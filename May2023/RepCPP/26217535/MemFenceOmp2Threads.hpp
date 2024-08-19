

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/mem/fence/Traits.hpp>

namespace alpaka
{
class MemFenceOmp2Threads : public concepts::Implements<ConceptMemFence, MemFenceOmp2Threads>
{
};

namespace trait
{
template<typename TMemScope>
struct MemFence<MemFenceOmp2Threads, TMemScope>
{
static auto mem_fence(MemFenceOmp2Threads const&, TMemScope const&)
{

#    pragma omp flush
}
};
} 
} 
#endif
