

#ifndef SYCL_SYCL_COMPAT_HPP
#define SYCL_SYCL_COMPAT_HPP

#include "sycl/sycl_utils.hpp"

#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_engine_base_t;

namespace compat {

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
const sycl_engine_base_t *sycl_engine,
const gpu::compute::binary_t &binary, const char *kernel_name);

void *get_native(const ::sycl::device &dev);
void *get_native(const ::sycl::context &ctx);

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
return reinterpret_cast<native_object_t>(get_native(sycl_object));
}

template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, int) -> decltype(cgh.host_task(f)) {
cgh.host_task(f);
}

template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, long)
-> decltype(cgh.codeplay_host_task(f)) {
cgh.codeplay_host_task(f);
}

template <typename H, typename F>
inline void host_task(H &cgh, F &&f) {
host_task_impl(cgh, f, 0);
}

uint64_t init_extensions(const ::sycl::device &dev);

constexpr auto target_device = ::sycl::target::device;

#if DNNL_USE_SYCL121_API
template <typename T, int dims>
using local_accessor = ::sycl::accessor<T, dims,
::sycl::access::mode::read_write, ::sycl::access::target::local>;

constexpr auto ext_intel_gpu_slices
= ::sycl::info::device::ext_intel_gpu_slices;
constexpr auto ext_intel_gpu_subslices_per_slice
= ::sycl::info::device::ext_intel_gpu_subslices_per_slice;

const auto cpu_selector_v = ::sycl::cpu_selector {};
const auto gpu_selector_v = ::sycl::gpu_selector {};
#else

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
template <typename T, int dims>
using local_accessor = ::sycl::accessor<T, dims,
::sycl::access::mode::read_write, ::sycl::access::target::local>;
#pragma clang diagnostic pop

using ext_intel_gpu_slices = ::sycl::ext::intel::info::device::gpu_slices;
using ext_intel_gpu_subslices_per_slice
= ::sycl::ext::intel::info::device::gpu_subslices_per_slice;

inline const auto &cpu_selector_v = ::sycl::cpu_selector_v;
inline const auto &gpu_selector_v = ::sycl::gpu_selector_v;

#endif

} 
} 
} 
} 

#endif
