


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>

#    include <string>

namespace alpaka
{
namespace detail
{
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
#    endif
struct IntelGpuSelector : sycl::device_selector
{
auto operator()(sycl::device const& dev) const -> int override
{
auto const vendor = dev.get_info<sycl::info::device::vendor>();
auto const is_intel_gpu = (vendor.find("Intel(R) Corporation") != std::string::npos) && dev.is_gpu();

return is_intel_gpu ? 1 : -1;
}
};
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
} 

using PltfGpuSyclIntel = PltfGenericSycl<detail::IntelGpuSelector>;
} 

namespace alpaka::trait
{
template<>
struct DevType<PltfGpuSyclIntel>
{
using type = DevGenericSycl<PltfGpuSyclIntel>; 
};
} 

#endif
