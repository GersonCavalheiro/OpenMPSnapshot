

#pragma once

#include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
template<typename TDev>
class EventGenericThreads;

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif

template<typename TDev>
class IGenericThreadsQueue
{
public:
virtual void enqueue(EventGenericThreads<TDev>&) = 0;
virtual void wait(EventGenericThreads<TDev> const&) = 0;
virtual ~IGenericThreadsQueue() = default;
};
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
} 
