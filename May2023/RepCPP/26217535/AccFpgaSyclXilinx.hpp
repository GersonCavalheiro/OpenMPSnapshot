

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/acc/AccGenericSycl.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/kernel/TaskKernelFpgaSyclXilinx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/PltfFpgaSyclXilinx.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <string>
#    include <utility>

namespace alpaka
{
template<typename TDim, typename TIdx>
class AccFpgaSyclXilinx final
: public AccGenericSycl<TDim, TIdx>
, public concepts::Implements<ConceptAcc, AccFpgaSyclXilinx<TDim, TIdx>>
{
public:
using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
};
} 

namespace alpaka::trait
{
template<typename TDim, typename TIdx>
struct GetAccName<AccFpgaSyclXilinx<TDim, TIdx>>
{
static auto getAccName() -> std::string
{
return "AccFpgaSyclXilinx<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
}
};

template<typename TDim, typename TIdx>
struct DevType<AccFpgaSyclXilinx<TDim, TIdx>>
{
using type = DevFpgaSyclXilinx;
};

template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
struct CreateTaskKernel<AccFpgaSyclXilinx<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
{
static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
{
return TaskKernelFpgaSyclXilinx<TDim, TIdx, TKernelFnObj, TArgs...>{
workDiv,
kernelFnObj,
std::forward<TArgs>(args)...};
}
};

template<typename TDim, typename TIdx>
struct PltfType<AccFpgaSyclXilinx<TDim, TIdx>>
{
using type = PltfFpgaSyclXilinx;
};

template<typename TDim, typename TIdx>
struct AccToTag<alpaka::AccFpgaSyclXilinx<TDim, TIdx>>
{
using type = alpaka::TagFpgaSyclXilinx;
};

template<typename TDim, typename TIdx>
struct TagToAcc<alpaka::TagFpgaSyclXilinx, TDim, TIdx>
{
using type = alpaka::AccFpgaSyclXilinx<TDim, TIdx>;
};
} 

#endif