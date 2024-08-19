

#pragma once

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/meta/InheritFromList.hpp>
#include <alpaka/meta/Unique.hpp>

#include <tuple>

namespace alpaka
{
template<typename TGridAtomic, typename TBlockAtomic, typename TThreadAtomic>
using AtomicHierarchy = alpaka::meta::InheritFromList<alpaka::meta::Unique<std::tuple<
TGridAtomic,
TBlockAtomic,
TThreadAtomic,
concepts::Implements<ConceptAtomicGrids, TGridAtomic>,
concepts::Implements<ConceptAtomicBlocks, TBlockAtomic>,
concepts::Implements<ConceptAtomicThreads, TThreadAtomic>>>>;
} 
