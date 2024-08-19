

#pragma once

#ifdef _OPENMP

#    include <alpaka/block/sync/Traits.hpp>
#    include <alpaka/core/Common.hpp>

namespace alpaka
{
class BlockSyncBarrierOmp : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOmp>
{
public:
std::uint8_t mutable m_generation = 0u;
int mutable m_result[2];
};

namespace trait
{
template<>
struct SyncBlockThreads<BlockSyncBarrierOmp>
{
ALPAKA_FN_HOST static auto syncBlockThreads(BlockSyncBarrierOmp const& ) -> void
{
#    pragma omp barrier
}
};

namespace detail
{
template<typename TOp>
struct AtomicOp;
template<>
struct AtomicOp<BlockCount>
{
void operator()(int& result, bool value)
{
#    pragma omp atomic
result += static_cast<int>(value);
}
};
template<>
struct AtomicOp<BlockAnd>
{
void operator()(int& result, bool value)
{
#    pragma omp atomic
result &= static_cast<int>(value);
}
};
template<>
struct AtomicOp<BlockOr>
{
void operator()(int& result, bool value)
{
#    pragma omp atomic
result |= static_cast<int>(value);
}
};
} 

template<typename TOp>
struct SyncBlockThreadsPredicate<TOp, BlockSyncBarrierOmp>
{
ALPAKA_NO_HOST_ACC_WARNING
ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncBarrierOmp const& blockSync, int predicate)
-> int
{
#    pragma omp single
{
++blockSync.m_generation;
blockSync.m_result[blockSync.m_generation % 2u] = TOp::InitialValue;
}

auto const generationMod2(blockSync.m_generation % 2u);
int& result(blockSync.m_result[generationMod2]);
bool const predicateBool(predicate != 0);

detail::AtomicOp<TOp>()(result, predicateBool);

#    pragma omp barrier

return blockSync.m_result[generationMod2];
}
};
} 
} 

#endif
