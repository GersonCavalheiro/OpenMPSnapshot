

#pragma once

#include <alpaka/rand/Philox/PhiloxStateless.hpp>
#include <alpaka/rand/Philox/PhiloxStatelessVector.hpp>
#include <alpaka/rand/Traits.hpp>

namespace alpaka::rand
{

template<typename TAcc>
class PhiloxStateless4x32x10Vector
: public alpaka::rand::engine::PhiloxStatelessVector<TAcc, engine::PhiloxParams<4, 32, 10>>
, public concepts::Implements<ConceptRand, PhiloxStateless4x32x10Vector<TAcc>>
{
public:
using EngineParams = engine::PhiloxParams<4, 32, 10>;
};
} 
