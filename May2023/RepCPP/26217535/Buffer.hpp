

#pragma once

#include "Defines.hpp"

#include <alpaka/test/acc/TestAccs.hpp>

#include <ostream>

namespace alpaka
{
namespace test
{
namespace unit
{
namespace math
{
template<typename TAcc, typename TData, size_t Tcapacity>
struct Buffer
{
using value_type = TData;
static constexpr size_t capacity = Tcapacity;
using Dim = typename alpaka::trait::DimType<TAcc>::type;
using Idx = typename alpaka::trait::IdxType<TAcc>::type;

using DevHost = alpaka::DevCpu;
using PltfHost = alpaka::Pltf<DevHost>;
using BufHost = alpaka::Buf<DevHost, TData, Dim, Idx>;

using DevAcc = alpaka::Dev<TAcc>;
using PltfAcc = alpaka::Pltf<DevAcc>;
using BufAcc = alpaka::Buf<DevAcc, TData, Dim, Idx>;

DevHost devHost;

BufHost hostBuffer;
BufAcc devBuffer;

TData* const pHostBuffer;
TData* const pDevBuffer;

Buffer() = delete;

Buffer(DevAcc const& devAcc)
: devHost{alpaka::getDevByIdx<PltfHost>(0u)}
, hostBuffer{alpaka::allocMappedBufIfSupported<PltfAcc, TData, Idx>(devHost, Tcapacity)}
, devBuffer{alpaka::allocBuf<TData, Idx>(devAcc, Tcapacity)}
, pHostBuffer{alpaka::getPtrNative(hostBuffer)}
, pDevBuffer{alpaka::getPtrNative(devBuffer)}
{
}

template<typename Queue>
auto copyToDevice(Queue queue) -> void
{
alpaka::memcpy(queue, devBuffer, hostBuffer);
}

template<typename Queue>
auto copyFromDevice(Queue queue) -> void
{
alpaka::memcpy(queue, hostBuffer, devBuffer);
}

ALPAKA_FN_ACC
auto operator()(size_t idx, TAcc const& ) const -> TData&
{
return pDevBuffer[idx];
}

ALPAKA_FN_HOST
auto operator()(size_t idx) const -> TData&
{
return pHostBuffer[idx];
}

ALPAKA_FN_HOST
friend auto operator<<(std::ostream& os, Buffer const& buffer) -> std::ostream&
{
os << "capacity: " << capacity << "\n";
for(size_t i = 0; i < capacity; ++i)
{
os << i << ": " << buffer.pHostBuffer[i] << "\n";
}
return os;
}
};

} 
} 
} 
} 
