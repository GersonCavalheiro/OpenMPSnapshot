

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/pltf/PltfFpgaSyclXilinx.hpp>

namespace alpaka
{
using DevFpgaSyclXilinx = DevGenericSycl<PltfFpgaSyclXilinx>;
}

#endif
