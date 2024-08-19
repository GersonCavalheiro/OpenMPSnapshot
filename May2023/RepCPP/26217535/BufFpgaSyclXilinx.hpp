


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/mem/buf/BufGenericSycl.hpp>

namespace alpaka
{
template<typename TElem, typename TDim, typename TIdx>
using BufFpgaSyclXilinx = BufGenericSycl<TElem, TDim, TIdx, DevFpgaSyclXilinx>;
} 

#endif
