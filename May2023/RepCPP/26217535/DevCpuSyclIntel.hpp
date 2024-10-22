

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/pltf/PltfCpuSyclIntel.hpp>

namespace alpaka
{
using DevCpuSyclIntel = DevGenericSycl<PltfCpuSyclIntel>;
} 

#endif
