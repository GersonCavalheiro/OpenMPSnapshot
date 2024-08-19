

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>
#    include <sycl/ext/intel/fpga_extensions.hpp>

namespace alpaka
{
#    ifdef ALPAKA_FPGA_EMULATION
using PltfFpgaSyclIntel = PltfGenericSycl<sycl::ext::intel::fpga_emulator_selector>;
#    else
using PltfFpgaSyclIntel = PltfGenericSycl<sycl::ext::intel::fpga_selector>;
#    endif
} 

namespace alpaka::trait
{
template<>
struct DevType<PltfFpgaSyclIntel>
{
using type = DevGenericSycl<PltfFpgaSyclIntel>; 
};
} 

#endif
