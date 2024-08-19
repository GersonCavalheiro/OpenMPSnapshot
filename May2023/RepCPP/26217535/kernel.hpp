

#pragma once

#include <alpaka/alpaka.hpp>

template<typename T, uint64_t size>
struct cheapArray
{
T data[size];

ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) -> T&
{
return data[index];
}

ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) const -> const T&
{
return data[index];
}
};

template<uint32_t TBlockSize, typename T, typename TFunc>
struct ReduceKernel
{
ALPAKA_NO_HOST_ACC_WARNING

template<typename TAcc, typename TElem, typename TIdx>
ALPAKA_FN_ACC auto operator()(
TAcc const& acc,
TElem const* const source,
TElem* destination,
TIdx const& n,
TFunc func) const -> void
{
auto& sdata(alpaka::declareSharedVar<cheapArray<T, TBlockSize>, __COUNTER__>(acc));

auto const blockIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));
auto const threadIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]));
auto const gridDimension(static_cast<uint32_t>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));

auto const linearizedIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]));

typename GetIterator<T, TElem, TAcc>::Iterator it(acc, source, linearizedIndex, gridDimension * TBlockSize, n);

T result = 0; 

if(threadIndex < n)
result = *(it++); 



while(it + 3 < it.end())
{
result = func(func(func(result, func(*it, *(it + 1))), *(it + 2)), *(it + 3));
it += 4;
}

while(it < it.end())
result = func(result, *(it++));

if(threadIndex < n)
sdata[threadIndex] = result;

alpaka::syncBlockThreads(acc);


ALPAKA_UNROLL()
for(uint32_t currentBlockSize = TBlockSize,
currentBlockSizeUp = (TBlockSize + 1) / 2; 
currentBlockSize > 1;
currentBlockSize = currentBlockSize / 2,
currentBlockSizeUp = (currentBlockSize + 1) / 2) 
{
bool cond = threadIndex < currentBlockSizeUp 
&& (threadIndex + currentBlockSizeUp) < TBlockSize 
&& (blockIndex * TBlockSize + threadIndex + currentBlockSizeUp) < n
&& threadIndex < n; 

if(cond)
sdata[threadIndex] = func(sdata[threadIndex], sdata[threadIndex + currentBlockSizeUp]);

alpaka::syncBlockThreads(acc);
}

if(threadIndex == 0 && threadIndex < n)
destination[blockIndex] = sdata[0];
}
};
