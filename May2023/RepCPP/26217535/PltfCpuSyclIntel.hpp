

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

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
struct IntelCpuSelector final : sycl::device_selector
{
auto operator()(sycl::device const& dev) const -> int override
{
auto const vendor = dev.get_info<sycl::info::device::vendor>();
auto const is_intel_cpu = (vendor.find("Intel(R) Corporation") != std::string::npos) && dev.is_cpu();

return is_intel_cpu ? 1 : -1;
}
};
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
} 

using PltfCpuSyclIntel = PltfGenericSycl<detail::IntelCpuSelector>;
} 

namespace alpaka::trait
{
template<>
struct DevType<PltfCpuSyclIntel>
{
using type = DevGenericSycl<PltfCpuSyclIntel>; 
};
} 

#endif
