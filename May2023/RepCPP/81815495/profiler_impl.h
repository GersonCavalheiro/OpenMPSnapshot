
#pragma once

#include "utilities/profiler.h" 

#include <string>
#include <chrono>


namespace Kratos::Internals {


template <class T>
Profiler<T>::Scope::Scope(Profiler::Item& rItem)
: Scope(rItem, Clock::now())
{
}


template <class T>
Profiler<T>::Scope::Scope(Profiler::Item& rItem, std::chrono::high_resolution_clock::time_point Begin)
: mrItem(rItem),
mBegin(Begin)
{
++mrItem.mCallCount;
++mrItem.mRecursionLevel;
}


template <class T>
Profiler<T>::Scope::~Scope()
{
if (!--mrItem.mRecursionLevel) {
const auto duration = std::chrono::duration_cast<Profiler::TimeUnit>(Clock::now() - mBegin);
mrItem.mCumulative += duration;
mrItem.mMin = std::min(mrItem.mMin, duration);
mrItem.mMax = std::max(mrItem.mMax, duration);
}
}


template <class T>
typename Profiler<T>::Scope Profiler<T>::Profile(Item& rItem)
{
return Scope(rItem);
}


} 
