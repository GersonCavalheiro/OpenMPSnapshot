

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

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
struct XilinxFpgaSelector : sycl::device_selector
{
auto operator()(sycl::device const& dev) const -> int override
{
auto const vendor = dev.get_info<sycl::info::device::vendor>();
auto const is_xilinx = (vendor.find("Xilinx") != std::string::npos);

return is_xilinx ? 1 : -1;
}
};
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
} 

using PltfFgpaSyclIntel = PltfGenericSycl<detail::XilinxFpgaSelector>;
} 

namespace alpaka::trait
{
template<>
struct DevType<PltfFpgaSyclXilinx>
{
using type = DevGenericSycl<PltfFpgaSyclXilinx>; 
};
} 

#endif
