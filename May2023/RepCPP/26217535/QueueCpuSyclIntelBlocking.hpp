

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <alpaka/dev/DevCpuSyclIntel.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>

namespace alpaka
{
using QueueCpuSyclIntelBlocking = QueueGenericSyclBlocking<DevCpuSyclIntel>;
}

#endif
