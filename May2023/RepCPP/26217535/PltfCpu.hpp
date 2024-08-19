

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <sstream>
#include <vector>

namespace alpaka
{
class PltfCpu : public concepts::Implements<ConceptPltf, PltfCpu>
{
public:
ALPAKA_FN_HOST PltfCpu() = delete;
};

namespace trait
{
template<>
struct DevType<PltfCpu>
{
using type = DevCpu;
};

template<>
struct GetDevCount<PltfCpu>
{
ALPAKA_FN_HOST static auto getDevCount() -> std::size_t
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

return 1;
}
};

template<>
struct GetDevByIdx<PltfCpu>
{
ALPAKA_FN_HOST static auto getDevByIdx(std::size_t const& devIdx) -> DevCpu
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

std::size_t const devCount(getDevCount<PltfCpu>());
if(devIdx >= devCount)
{
std::stringstream ssErr;
ssErr << "Unable to return device handle for CPU device with index " << devIdx
<< " because there are only " << devCount << " devices!";
throw std::runtime_error(ssErr.str());
}

return {};
}
};
} 
} 