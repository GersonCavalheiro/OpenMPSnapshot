

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/pltf/PltfGpuSyclIntel.hpp>

namespace alpaka
{
using DevGpuSyclIntel = DevGenericSycl<PltfGpuSyclIntel>;
}

#endif
