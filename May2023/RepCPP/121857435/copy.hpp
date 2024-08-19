#pragma once

#include "defs.hpp"
#include <cuda_runtime.h>
#include <type_traits>

namespace bnmf_algs {
namespace cuda {


template <typename DstMemory, typename SrcMemory,
template <typename> class HostMemoryBase,
template <typename> class DeviceMemoryBase>
constexpr cudaMemcpyKind infer_copy_kind() {
typedef typename DstMemory::value_type DstT;
typedef typename SrcMemory::value_type SrcT;

typedef typename std::remove_cv<DstMemory>::type DstType;
typedef typename std::remove_cv<SrcMemory>::type SrcType;

return
(std::is_same<DstType, HostMemoryBase<DstT>>::value &&
std::is_same<SrcType, HostMemoryBase<SrcT>>::value)
? cudaMemcpyKind::cudaMemcpyHostToHost
:

(std::is_same<DstType, HostMemoryBase<DstT>>::value &&
std::is_same<SrcType, DeviceMemoryBase<SrcT>>::value)
? cudaMemcpyKind::cudaMemcpyDeviceToHost
:

(std::is_same<DstType, DeviceMemoryBase<DstT>>::value &&
std::is_same<SrcType, HostMemoryBase<SrcT>>::value)
? cudaMemcpyKind::cudaMemcpyHostToDevice
:

(std::is_same<DstType, DeviceMemoryBase<DstT>>::value &&
std::is_same<SrcType, DeviceMemoryBase<SrcT>>::value)
? cudaMemcpyKind::cudaMemcpyDeviceToDevice
: cudaMemcpyKind::cudaMemcpyDefault;
}


template <typename DstMemory1D, typename SrcMemory1D>
void copy1D(DstMemory1D& destination, const SrcMemory1D& source) {
static constexpr cudaMemcpyKind kind =
infer_copy_kind<DstMemory1D, SrcMemory1D, HostMemory1D,
DeviceMemory1D>();
static_assert(kind != cudaMemcpyDefault,
"Invalid copy direction in cuda::copy1D");

auto err =
cudaMemcpy(destination.data(), source.data(), source.bytes(), kind);
BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy1D");
}


template <typename DstPitchedMemory2D, typename SrcPitchedMemory2D>
void copy2D(DstPitchedMemory2D& destination, const SrcPitchedMemory2D& source) {
static constexpr cudaMemcpyKind kind =
infer_copy_kind<DstPitchedMemory2D, SrcPitchedMemory2D, HostMemory2D,
DeviceMemory2D>();
static_assert(kind != cudaMemcpyDefault,
"Invalid copy direction in cuda::copy2D");

auto err =
cudaMemcpy2D(destination.data(), destination.pitch(), source.data(),
source.pitch(), source.width(), source.height(), kind);
BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy2D");
}


template <typename DstPitchedMemory3D, typename SrcPitchedMemory3D>
void copy3D(DstPitchedMemory3D& destination, const SrcPitchedMemory3D& source) {
static constexpr cudaMemcpyKind kind =
infer_copy_kind<DstPitchedMemory3D, SrcPitchedMemory3D, HostMemory3D,
DeviceMemory3D>();
static_assert(kind != cudaMemcpyDefault,
"Invalid copy direction in cuda::copy3D");

cudaMemcpy3DParms params = {nullptr};
params.srcPtr = source.pitched_ptr();
params.dstPtr = destination.pitched_ptr();
params.extent = source.extent();
params.kind = kind;

auto err = cudaMemcpy3D(&params);
BNMF_ASSERT(err == cudaSuccess, "Error copying memory in cuda::copy3D");
}
} 
} 
