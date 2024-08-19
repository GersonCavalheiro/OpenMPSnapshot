

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstddef>

namespace alpaka::test
{
template<typename TIdx>
struct CreateVecWithIdx
{
template<std::size_t Tidx>
struct ForExtentBuf
{
ALPAKA_FN_HOST_ACC static auto create()
{
return static_cast<TIdx>(11u - Tidx);
}
};

template<std::size_t Tidx>
struct ForExtentSubView
{
ALPAKA_FN_HOST_ACC static auto create()
{
return static_cast<TIdx>(8u - (Tidx * 2u));
}
};

template<std::size_t Tidx>
struct ForOffset
{
ALPAKA_FN_HOST_ACC static auto create()
{
return static_cast<TIdx>(2u + Tidx);
}
};
};
} 
