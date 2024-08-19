#pragma once

#include <vector>
#include <omp.h>
#include <array>

namespace SearchAlgorithms
{

constexpr unsigned MAX_THREAD_COUNT = 64;
using Shbool = short;	


template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT ShorterParallelSearch(const std::vector<VecType>& sortedVec, const VecType& val)
{
const unsigned threadCount = omp_get_max_threads();
const unsigned lastThrdID = threadCount - 1;

std::array<Shbool, MAX_THREAD_COUNT+1> isValToTheRight; 
isValToTheRight[threadCount] = false;	

VecSizeT l = 0;	
VecSizeT r = sortedVec.size() - 1; 
VecSizeT indexFound = -1;

#pragma omp parallel \
shared(sortedVec, val, l, r, isValToTheRight, indexFound) \
default(none)	
{
const unsigned thrdID = omp_get_thread_num();

Shbool& currThrdSaysItsRight = isValToTheRight[thrdID];
const Shbool& nextThrdSaysItsRight = isValToTheRight[thrdID + 1];

while (l <= r - threadCount)
{
const VecSizeT segmSize = (r - l + 1) / (threadCount + 1) + 1;
const VecSizeT currThrdPos = l + segmSize * thrdID;
currThrdSaysItsRight = (sortedVec[currThrdPos] <= val);

#pragma omp barrier

if (currThrdSaysItsRight != nextThrdSaysItsRight)
{
l = currThrdPos;
if (thrdID != lastThrdID)
r = currThrdPos + segmSize - 1;
}

#pragma omp barrier
}

const VecSizeT currThrdPos = l + thrdID;
if (currThrdPos <= r && sortedVec[currThrdPos] == val)
indexFound = currThrdPos;
}

return indexFound;
}


template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT ParallelSearch(const std::vector<VecType>& sortedVec, const VecType& val)
{
const unsigned threadCount = omp_get_max_threads();
const unsigned firstThrdID = 0;

std::array<Shbool, MAX_THREAD_COUNT + 1> isValToTheRight; 
isValToTheRight[0] = true;	

std::array<VecSizeT, MAX_THREAD_COUNT> thrdsPos;	

VecSizeT l = 0;	
VecSizeT r = sortedVec.size() - 1; 

#pragma omp parallel \
shared(l, r, thrdsPos, isValToTheRight, sortedVec, val) \
default(none)	
{
const unsigned thrdID = omp_get_thread_num();
VecSizeT& currThrdPos = thrdsPos[thrdID];
Shbool& currThrdSaysItsRight = isValToTheRight[thrdID + 1];
const Shbool& prevThrdSaysItsRight = isValToTheRight[thrdID];

while (l < r)
{
const VecSizeT segmSize = (r - l + 1) / (threadCount + 1) + 1;
currThrdPos = l + segmSize * (thrdID + 1) - 1;
currThrdSaysItsRight = (sortedVec[currThrdPos] <= val);

#pragma omp barrier

if (currThrdSaysItsRight != prevThrdSaysItsRight)
{
r = currThrdPos - 1;
if (thrdID != firstThrdID)
{
l = thrdsPos[thrdID - 1];
}
}

#pragma omp single
{
if (isValToTheRight[threadCount])
{
l = thrdsPos[threadCount-1];
}
}
}
}


if (sortedVec[l] != val)
return VecSizeT(-1);

return l;
}

template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT BinarySearch(const std::vector<VecType>& sortedVec, VecType val)
{
VecSizeT l = 0, r = sortedVec.size() - 1, m;
while (l < r)
{
m = l + (r - l) / 2 + 1;
if (sortedVec[m] <= val)
{
l = m;
}
else
{
r = m - 1;
}
}


if (sortedVec[l] != val)
return VecSizeT(-1);

return l;
}

template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT BinarySearchInParallel(const std::vector<VecType>& sortedVec, const VecType& val)
{
VecSizeT idxFound = -1;
const unsigned threadCount = omp_get_max_threads();
const VecSizeT vecSize = sortedVec.size();

if (vecSize <= threadCount)
{
#pragma omp parallel shared(idxFound)
{
const unsigned thrdID = omp_get_thread_num();
if (thrdID < vecSize && sortedVec[thrdID] == val)
idxFound = thrdID;
}

return idxFound;
}

#pragma omp parallel
{
const unsigned thrdID = omp_get_thread_num();
VecSizeT segmSize = vecSize / threadCount;
VecSizeT l = segmSize * thrdID;
VecSizeT r = segmSize * (thrdID + 1);

if (thrdID == threadCount - 1)	
{
segmSize += vecSize % threadCount;
l = vecSize - segmSize;
r = vecSize - 1;
}

while (l < r)
{
const VecSizeT m = l + (r - l) / 2 + 1;
if (sortedVec[m] <= val)
{
l = m;
}
else
{
r = m - 1;
}
}

if (sortedVec[l] == val)
idxFound = l;
}

return idxFound;
}

}

