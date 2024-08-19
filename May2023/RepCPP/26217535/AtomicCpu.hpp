

#pragma once

#include <boost/version.hpp>

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    define ALPAKA_DISABLE_ATOMIC_ATOMICREF
#endif

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    include <alpaka/atomic/AtomicAtomicRef.hpp>
#else
#    include <alpaka/atomic/AtomicStdLibLock.hpp>
#endif 

namespace alpaka
{
#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
using AtomicCpu = AtomicAtomicRef;
#else
using AtomicCpu = AtomicStdLibLock<16>;
#endif 

} 
