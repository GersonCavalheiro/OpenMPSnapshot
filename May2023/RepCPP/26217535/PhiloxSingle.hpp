

#pragma once

#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>
#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>

#include <utility>

namespace alpaka::rand::engine
{

template<typename TCounter, typename TKey>
struct PhiloxStateSingle
{
using Counter = TCounter;
using Key = TKey;

Counter counter; 
Key key; 
Counter result; 
std::uint32_t position; 
};


template<typename TAcc, typename TParams>
class PhiloxSingle : public trait::PhiloxBaseTraits<TAcc, TParams, PhiloxSingle<TAcc, TParams>>::Base
{
public:
using Traits = typename trait::PhiloxBaseTraits<TAcc, TParams, PhiloxSingle<TAcc, TParams>>;

using Counter = typename Traits::Counter; 
using Key = typename Traits::Key; 
using State = PhiloxStateSingle<Counter, Key>; 

State state; 

protected:

ALPAKA_FN_HOST_ACC void advanceState()
{
this->advanceCounter(state.counter);
state.result = this->nRounds(state.counter, state.key);
state.position = 0;
}


ALPAKA_FN_HOST_ACC auto nextNumber()
{
auto result = state.result[0];
state.position++;
if(state.position == TParams::counterSize)
{
advanceState();
}
else
{
state.result[0] = state.result[1];
state.result[1] = state.result[2];
state.result[2] = state.result[3];
}

return result;
}

ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
{
static_assert(TParams::counterSize == 4, "Only counterSize is supported.");
state.position = static_cast<decltype(state.position)>(state.position + (offset & 3));
offset += state.position < 4 ? 0 : 4;
state.position -= state.position < 4 ? 0 : 4u;
for(auto numShifts = state.position; numShifts > 0; --numShifts)
{
state.result[0] = state.result[1];
state.result[1] = state.result[2];
state.result[2] = state.result[3];
}
this->skip4(offset / 4);
}

public:

ALPAKA_FN_HOST_ACC PhiloxSingle(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
: state{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}, {0, 0, 0, 0}, 0}
{
this->skipSubsequence(subsequence);
skip(offset);
advanceState();
}


ALPAKA_FN_HOST_ACC auto operator()()
{
return nextNumber();
}
};
} 
