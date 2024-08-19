

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/dev/DevGpuSyclIntel.hpp>
#    include <alpaka/event/EventGenericSycl.hpp>

namespace alpaka
{
using EventGpuSyclIntel = EventGenericSycl<DevGpuSyclIntel>;
} 

#endif
