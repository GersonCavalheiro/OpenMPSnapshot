

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/test/idx/TestIdxs.hpp>

#include <iosfwd>
#include <tuple>
#include <type_traits>

#if defined(ALPAKA_CI)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA                                                       \
|| defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
#        define ALPAKA_CUDA_CI
#    endif
#endif

namespace alpaka::test
{
namespace detail
{
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
template<typename TDim, typename TIdx>
using AccCpuSerialIfAvailableElseInt = AccCpuSerial<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuSerialIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) && !defined(ALPAKA_CUDA_CI)
template<typename TDim, typename TIdx>
using AccCpuThreadsIfAvailableElseInt = AccCpuThreads<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
template<typename TDim, typename TIdx>
using AccCpuTbbIfAvailableElseInt = AccCpuTbbBlocks<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuTbbIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
template<typename TDim, typename TIdx>
using AccCpuOmp2BlocksIfAvailableElseInt = AccCpuOmp2Blocks<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuOmp2BlocksIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) && !defined(ALPAKA_CUDA_CI)
template<typename TDim, typename TIdx>
using AccCpuOmp2ThreadsIfAvailableElseInt = AccCpuOmp2Threads<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuOmp2ThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (BOOST_LANG_CUDA || defined(ALPAKA_HOST_ONLY))
template<typename TDim, typename TIdx>
using AccGpuCudaRtIfAvailableElseInt = AccGpuCudaRt<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccGpuCudaRtIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && (BOOST_LANG_HIP || defined(ALPAKA_HOST_ONLY))
template<typename TDim, typename TIdx>
using AccGpuHipRtIfAvailableElseInt =
typename std::conditional<std::is_same_v<TDim, DimInt<3u>> == false, AccGpuHipRt<TDim, TIdx>, int>::type;
#else
template<typename TDim, typename TIdx>
using AccGpuHipRtIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_CPU)
template<typename TDim, typename TIdx>
using AccCpuSyclIntelIfAvailableElseInt = alpaka::AccCpuSyclIntel<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccCpuSyclIntelIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_FPGA)
template<typename TDim, typename TIdx>
using AccFpgaSyclIntelIfAvailableElseInt = alpaka::AccFpgaSyclIntel<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccFpgaSyclIntelIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)
template<typename TDim, typename TIdx>
using AccFpgaSyclXilinxIfAvailableElseInt = alpaka::AccFpgaSyclXilinx<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccFpgaSyclXilinxIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_GPU)
template<typename TDim, typename TIdx>
using AccGpuSyclIntelIfAvailableElseInt = alpaka::AccGpuSyclIntel<TDim, TIdx>;
#else
template<typename TDim, typename TIdx>
using AccGpuSyclIntelIfAvailableElseInt = int;
#endif

template<typename TDim, typename TIdx>
using EnabledAccsElseInt = std::tuple<
AccCpuSerialIfAvailableElseInt<TDim, TIdx>,
AccCpuThreadsIfAvailableElseInt<TDim, TIdx>,
AccCpuTbbIfAvailableElseInt<TDim, TIdx>,
AccCpuOmp2BlocksIfAvailableElseInt<TDim, TIdx>,
AccCpuOmp2ThreadsIfAvailableElseInt<TDim, TIdx>,
AccGpuCudaRtIfAvailableElseInt<TDim, TIdx>,
AccGpuHipRtIfAvailableElseInt<TDim, TIdx>,
AccCpuSyclIntelIfAvailableElseInt<TDim, TIdx>,
AccFpgaSyclIntelIfAvailableElseInt<TDim, TIdx>,
AccFpgaSyclXilinxIfAvailableElseInt<TDim, TIdx>,
AccGpuSyclIntelIfAvailableElseInt<TDim, TIdx>>;
} 

template<typename TDim, typename TIdx>
using EnabledAccs = typename meta::Filter<detail::EnabledAccsElseInt<TDim, TIdx>, std::is_class>;

namespace detail
{
struct StreamOutAccName
{
template<typename TAcc>
ALPAKA_FN_HOST auto operator()(std::ostream& os) -> void
{
os << getAccName<TAcc>();
os << " ";
}
};
} 

template<typename TDim, typename TIdx>
ALPAKA_FN_HOST auto writeEnabledAccs(std::ostream& os) -> void
{
os << "Accelerators enabled: ";

meta::forEachType<EnabledAccs<TDim, TIdx>>(detail::StreamOutAccName(), std::ref(os));

os << std::endl;
}

namespace detail
{
using TestDimIdxTuples = meta::CartesianProduct<std::tuple, TestDims, TestIdxs>;

template<typename TList>
using ApplyEnabledAccs = meta::Apply<TList, EnabledAccs>;

using InstantiatedEnabledAccs = meta::Transform<TestDimIdxTuples, ApplyEnabledAccs>;
} 

using TestAccs = meta::Apply<detail::InstantiatedEnabledAccs, meta::Concatenate>;
} 
