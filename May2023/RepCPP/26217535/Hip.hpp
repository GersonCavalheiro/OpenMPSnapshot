

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    if !BOOST_LANG_HIP && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/core/CudaHipCommon.hpp>
#    include <alpaka/core/UniformCudaHip.hpp>

#endif
