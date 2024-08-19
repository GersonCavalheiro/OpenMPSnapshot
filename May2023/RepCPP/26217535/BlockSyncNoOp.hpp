

#pragma once

#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/core/Common.hpp>

namespace alpaka
{
class BlockSyncNoOp : public concepts::Implements<ConceptBlockSync, BlockSyncNoOp>
{
};

namespace trait
{
template<>
struct SyncBlockThreads<BlockSyncNoOp>
{
ALPAKA_NO_HOST_ACC_WARNING
ALPAKA_FN_ACC static auto syncBlockThreads(BlockSyncNoOp const& ) -> void
{
}
};

template<typename TOp>
struct SyncBlockThreadsPredicate<TOp, BlockSyncNoOp>
{
ALPAKA_NO_HOST_ACC_WARNING
ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncNoOp const& , int predicate)
-> int
{
return predicate;
}
};
} 
} 
