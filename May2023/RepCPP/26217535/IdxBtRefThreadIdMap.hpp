

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <map>
#    include <thread>

namespace alpaka
{
namespace bt
{
template<typename TDim, typename TIdx>
class IdxBtRefThreadIdMap : public concepts::Implements<ConceptIdxBt, IdxBtRefThreadIdMap<TDim, TIdx>>
{
public:
using ThreadIdToIdxMap = std::map<std::thread::id, Vec<TDim, TIdx>>;

ALPAKA_FN_HOST IdxBtRefThreadIdMap(ThreadIdToIdxMap const& mThreadToIndices)
: m_threadToIndexMap(mThreadToIndices)
{
}
ALPAKA_FN_HOST IdxBtRefThreadIdMap(IdxBtRefThreadIdMap const&) = delete;
ALPAKA_FN_HOST auto operator=(IdxBtRefThreadIdMap const&) -> IdxBtRefThreadIdMap& = delete;

public:
ThreadIdToIdxMap const& m_threadToIndexMap; 
};
} 

namespace trait
{
template<typename TDim, typename TIdx>
struct DimType<bt::IdxBtRefThreadIdMap<TDim, TIdx>>
{
using type = TDim;
};

template<typename TDim, typename TIdx>
struct GetIdx<bt::IdxBtRefThreadIdMap<TDim, TIdx>, origin::Block, unit::Threads>
{
template<typename TWorkDiv>
ALPAKA_FN_HOST static auto getIdx(
bt::IdxBtRefThreadIdMap<TDim, TIdx> const& idx,
TWorkDiv const& ) -> Vec<TDim, TIdx>
{
auto const threadId = std::this_thread::get_id();
auto const threadEntry = idx.m_threadToIndexMap.find(threadId);
ALPAKA_ASSERT(threadEntry != std::end(idx.m_threadToIndexMap));
return threadEntry->second;
}
};

template<typename TDim, typename TIdx>
struct IdxType<bt::IdxBtRefThreadIdMap<TDim, TIdx>>
{
using type = TIdx;
};
} 
} 

#endif
