

#pragma once

#include <alpaka/core/Unroll.hpp>
#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>
#include <alpaka/rand/Philox/PhiloxConstants.hpp>

#include <utility>


namespace alpaka::rand::engine
{

template<unsigned TCounterSize, unsigned TWidth, unsigned TRounds>
struct PhiloxParams
{
static constexpr unsigned counterSize = TCounterSize;
static constexpr unsigned width = TWidth;
static constexpr unsigned rounds = TRounds;
};


template<typename TBackend, typename TParams>
class PhiloxStateless : public PhiloxConstants<TParams>
{
static constexpr unsigned numRounds()
{
return TParams::rounds;
}
static constexpr unsigned vectorSize()
{
return TParams::counterSize;
}
static constexpr unsigned numberWidth()
{
return TParams::width;
}

static_assert(numRounds() > 0, "Number of Philox rounds must be > 0.");
static_assert(vectorSize() % 2 == 0, "Philox counter size must be an even number.");
static_assert(vectorSize() <= 16, "Philox SP network is not specified for sizes > 16.");
static_assert(numberWidth() % 8 == 0, "Philox number width in bits must be a multiple of 8.");

static_assert(numberWidth() == 32, "Philox implemented only for 32 bit numbers.");

public:
using Counter = typename TBackend::Counter;
using Key = typename TBackend::Key;
using Constants = PhiloxConstants<TParams>;

protected:

static ALPAKA_FN_HOST_ACC auto singleRound(Counter const& counter, Key const& key)
{
std::uint32_t H0, L0, H1, L1;
multiplyAndSplit64to32(counter[0], Constants::MULTIPLITER_4x32_0(), H0, L0);
multiplyAndSplit64to32(counter[2], Constants::MULTIPLITER_4x32_1(), H1, L1);
return Counter{H1 ^ counter[1] ^ key[0], L1, H0 ^ counter[3] ^ key[1], L0};
}


static ALPAKA_FN_HOST_ACC auto bumpKey(Key const& key)
{
return Key{key[0] + Constants::WEYL_32_0(), key[1] + Constants::WEYL_32_1()};
}


static ALPAKA_FN_HOST_ACC auto nRounds(Counter const& counter_in, Key const& key_in) -> Counter
{
Key key{key_in};
Counter counter = singleRound(counter_in, key);

ALPAKA_UNROLL(numRounds())
for(unsigned int n = 0; n < numRounds(); ++n)
{
key = bumpKey(key);
counter = singleRound(counter, key);
}

return counter;
}

public:

static ALPAKA_FN_HOST_ACC auto generate(Counter const& counter, Key const& key) -> Counter
{
return nRounds(counter, key);
}
};
} 
