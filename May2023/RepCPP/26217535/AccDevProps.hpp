

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/vec/Vec.hpp>

#include <string>
#include <vector>

namespace alpaka
{
template<typename TDim, typename TIdx>
struct AccDevProps
{
static_assert(
sizeof(TIdx) >= sizeof(int),
"Index type is not supported, consider using int or a larger type.");
ALPAKA_FN_HOST AccDevProps(
TIdx const& multiProcessorCount,
Vec<TDim, TIdx> const& gridBlockExtentMax,
TIdx const& gridBlockCountMax,
Vec<TDim, TIdx> const& blockThreadExtentMax,
TIdx const& blockThreadCountMax,
Vec<TDim, TIdx> const& threadElemExtentMax,
TIdx const& threadElemCountMax,
size_t const& sharedMemSizeBytes)
: m_gridBlockExtentMax(gridBlockExtentMax)
, m_blockThreadExtentMax(blockThreadExtentMax)
, m_threadElemExtentMax(threadElemExtentMax)
, m_gridBlockCountMax(gridBlockCountMax)
, m_blockThreadCountMax(blockThreadCountMax)
, m_threadElemCountMax(threadElemCountMax)
, m_multiProcessorCount(multiProcessorCount)
, m_sharedMemSizeBytes(sharedMemSizeBytes)
{
}

Vec<TDim, TIdx> m_gridBlockExtentMax; 
Vec<TDim, TIdx> m_blockThreadExtentMax; 
Vec<TDim, TIdx> m_threadElemExtentMax; 

TIdx m_gridBlockCountMax; 
TIdx m_blockThreadCountMax; 
TIdx m_threadElemCountMax; 

TIdx m_multiProcessorCount; 
size_t m_sharedMemSizeBytes; 
};
} 
