

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <alpaka/dev/DevCpuSyclIntel.hpp>
#    include <alpaka/event/EventGenericSycl.hpp>

namespace alpaka
{
using EventCpuSyclIntel = EventGenericSycl<DevCpuSyclIntel>;
} 

#endif
