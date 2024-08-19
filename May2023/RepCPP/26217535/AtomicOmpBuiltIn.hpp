

#pragma once

#ifdef _OPENMP

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/atomic/Traits.hpp>
#    include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
class AtomicOmpBuiltIn
{
};

namespace trait
{
#    if _OPENMP >= 201107

template<typename T, typename THierarchy>
struct AtomicOp<AtomicAdd, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture
{
old = ref;
ref += value;
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicSub, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture
{
old = ref;
ref -= value;
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicExch, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        pragma omp atomic capture
{
old = ref;
ref = value;
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicAnd, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture
{
old = ref;
ref &= value;
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicOr, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture
{
old = ref;
ref |= value;
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicXor, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture
{
old = ref;
ref ^= value;
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

#    endif 

#    if _OPENMP >= 202011

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMin, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        pragma omp atomic capture compare
{
old = ref;
ref = (ref <= value) ? ref : value;
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMax, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        pragma omp atomic capture compare
{
old = ref;
ref = (ref >= value) ? ref : value;
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicInc, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture compare
{
old = ref;
ref = ((ref >= value) ? 0 : (ref + 1));
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicDec, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
auto& ref(*addr);
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
#        pragma omp atomic capture compare
{
old = ref;
ref = ((ref == 0) || (ref > value)) ? value : (ref - 1);
}
#        if BOOST_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicCas, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(
AtomicOmpBuiltIn const&,
T* const addr,
T const& compare,
T const& value) -> T
{
T old;
auto& ref(*addr);
#        pragma omp atomic capture compare
{
old = ref;
ref = (ref == compare ? value : ref);
}
return old;
}
};

#    else
template<typename TOp, typename T, typename THierarchy>
struct AtomicOp<TOp, AtomicOmpBuiltIn, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
{
T old;
#        pragma omp critical(AlpakaOmpAtomicOp)
{
old = TOp()(addr, value);
}
return old;
}
ALPAKA_FN_HOST static auto atomicOp(
AtomicOmpBuiltIn const&,
T* const addr,
T const& compare,
T const& value) -> T
{
T old;
#        pragma omp critical(AlpakaOmpAtomicOp2)
{
old = TOp()(addr, compare, value);
}
return old;
}
};

#    endif 

} 
} 

#endif
