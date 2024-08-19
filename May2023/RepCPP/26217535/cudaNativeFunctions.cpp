

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch_test_macros.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA

#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wmissing-prototypes"
#    endif
__device__ auto userDefinedThreadFence() -> void
{
__threadfence();
}
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

class CudaOnlyTestKernel
{
public:
template<typename TAcc>
ALPAKA_FN_ACC auto operator()(TAcc const& , bool* success) const -> void
{
__threadfence_block();
userDefinedThreadFence();
__threadfence_system();

*success = true;
}
};

TEST_CASE("cudaOnlyModeWorking", "[cudaOnly]")
{
using TAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::uint32_t>;
using Dim = alpaka::Dim<TAcc>;
using Idx = alpaka::Idx<TAcc>;

alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

CudaOnlyTestKernel kernel;

REQUIRE(fixture(kernel));
}

#endif
