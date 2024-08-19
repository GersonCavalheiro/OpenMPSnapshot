

#pragma once

#include <alpaka/warp/Traits.hpp>

#include <cstdint>

namespace alpaka::warp
{
class WarpSingleThread : public concepts::Implements<ConceptWarp, WarpSingleThread>
{
};

namespace trait
{
template<>
struct GetSize<WarpSingleThread>
{
static auto getSize(warp::WarpSingleThread const& )
{
return 1;
}
};

template<>
struct Activemask<WarpSingleThread>
{
static auto activemask(warp::WarpSingleThread const& )
{
return 1u;
}
};

template<>
struct All<WarpSingleThread>
{
static auto all(warp::WarpSingleThread const& , std::int32_t predicate)
{
return predicate;
}
};

template<>
struct Any<WarpSingleThread>
{
static auto any(warp::WarpSingleThread const& , std::int32_t predicate)
{
return predicate;
}
};

template<>
struct Ballot<WarpSingleThread>
{
static auto ballot(warp::WarpSingleThread const& , std::int32_t predicate)
{
return predicate ? 1u : 0u;
}
};

template<>
struct Shfl<WarpSingleThread>
{
static auto shfl(
warp::WarpSingleThread const& ,
std::int32_t val,
std::int32_t ,
std::int32_t )
{
return val;
}

static auto shfl(
warp::WarpSingleThread const& ,
float val,
std::int32_t ,
std::int32_t )
{
return val;
}
};
} 
} 
