

#pragma once

#include <alpaka/rand/Philox/PhiloxStateless.hpp>

#include <utility>


namespace alpaka::rand::engine
{

template<typename TBackend, typename TParams, typename TImpl>
class PhiloxBaseCommon
: public TBackend
, public PhiloxStateless<TBackend, TParams>
{
public:
using Counter = typename PhiloxStateless<TBackend, TParams>::Counter;
using Key = typename PhiloxStateless<TBackend, TParams>::Key;

protected:

ALPAKA_FN_HOST_ACC void advanceCounter(Counter& counter)
{
counter[0]++;

if(counter[0] == 0)
{
counter[1]++;
if(counter[1] == 0)
{
counter[2]++;
if(counter[2] == 0)
{
counter[3]++;
}
}
}
}


ALPAKA_FN_HOST_ACC void skip4(uint64_t offset)
{
Counter& counter = static_cast<TImpl*>(this)->state.counter;
Counter temp = counter;
counter[0] += low32Bits(offset);
counter[1] += high32Bits(offset) + (counter[0] < temp[0] ? 1 : 0);
counter[2] += (counter[0] < temp[1] ? 1u : 0u);
counter[3] += (counter[0] < temp[2] ? 1u : 0u);
}


ALPAKA_FN_HOST_ACC void skipSubsequence(uint64_t subsequence)
{
Counter& counter = static_cast<TImpl*>(this)->state.counter;
Counter temp = counter;
counter[2] += low32Bits(subsequence);
counter[3] += high32Bits(subsequence) + (counter[2] < temp[2] ? 1 : 0);
}
};
} 
