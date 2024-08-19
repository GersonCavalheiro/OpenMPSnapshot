

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>

#include <utility>


namespace alpaka::rand::engine
{

template<typename TAcc, typename TParams>
class PhiloxStatelessVector : public trait::PhiloxStatelessBaseTraits<TAcc, TParams>::Base
{
};
} 
