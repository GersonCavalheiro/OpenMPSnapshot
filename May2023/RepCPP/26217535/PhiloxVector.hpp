

#pragma once

#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>
#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>

#include <utility>


namespace alpaka::rand::engine
{

template<typename TCounter, typename TKey>
struct PhiloxStateVector
{
using Counter = TCounter;
using Key = TKey;

Counter counter; 
Key key; 
};


template<typename TAcc, typename TParams>
class PhiloxVector : public trait::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>::Base
{
public:
using Traits = trait::PhiloxBaseTraits<TAcc, TParams, PhiloxVector<TAcc, TParams>>;

using Counter = typename Traits::Counter; 
using Key = typename Traits::Key; 
using State = PhiloxStateVector<Counter, Key>; 
template<typename TDistributionResultScalar>
using ResultContainer = typename Traits::template ResultContainer<TDistributionResultScalar>;

State state;

protected:

ALPAKA_FN_HOST_ACC auto nextVector()
{
this->advanceCounter(state.counter);
return this->nRounds(state.counter, state.key);
}


ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
{
this->skip4(offset);
}

public:

ALPAKA_FN_HOST_ACC PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
: state{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}}
{
this->skipSubsequence(subsequence);
skip(offset);
nextVector();
}


ALPAKA_FN_HOST_ACC auto operator()()
{
return nextVector();
}
};
} 
