

#pragma once

namespace hydra_thrust
{

namespace random
{

namespace detail
{

struct random_core_access
{

template<typename OStream, typename EngineOrDistribution>
static OStream &stream_out(OStream &os, const EngineOrDistribution &x)
{
return x.stream_out(os);
}

template<typename IStream, typename EngineOrDistribution>
static IStream &stream_in(IStream &is, EngineOrDistribution &x)
{
return x.stream_in(is);
}

template<typename EngineOrDistribution>
__host__ __device__
static bool equal(const EngineOrDistribution &lhs, const EngineOrDistribution &rhs)
{
return lhs.equal(rhs);
}

}; 

} 

} 

} 

