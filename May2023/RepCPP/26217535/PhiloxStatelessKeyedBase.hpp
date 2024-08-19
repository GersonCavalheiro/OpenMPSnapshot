

#pragma once

#include <alpaka/rand/Philox/PhiloxStateless.hpp>

namespace alpaka::rand::engine
{

template<typename TBackend, typename TParams>
struct PhiloxStatelessKeyedBase : public PhiloxStateless<TBackend, TParams>
{
public:
using Counter = typename PhiloxStateless<TBackend, TParams>::Counter;
using Key = typename PhiloxStateless<TBackend, TParams>::Key;

const Key m_key;

PhiloxStatelessKeyedBase(Key&& key) : m_key(std::move(key))
{
}

ALPAKA_FN_HOST_ACC auto operator()(Counter const& counter) const
{
return this->generate(counter, m_key);
}
};
} 
