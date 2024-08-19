

#include "gpu/amd/sycl_hip_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

hip_sycl_scoped_context_handler_t::hip_sycl_scoped_context_handler_t(
const sycl_hip_engine_t &engine)
: need_to_recover_(false) {
try {
auto desired = engine.get_underlying_context();
HIP_EXECUTE_FUNC(hipCtxGetCurrent, &original_);

if (original_ != desired) {

HIP_EXECUTE_FUNC(hipCtxSetCurrent, desired);

need_to_recover_
= !(original_ == nullptr && engine.has_primary_context());
}
} catch (const std::runtime_error &e) {
error::wrap_c_api(status::runtime_error, e.what());
}
}

hip_sycl_scoped_context_handler_t::
~hip_sycl_scoped_context_handler_t() noexcept(false) {

try {
if (need_to_recover_) { HIP_EXECUTE_FUNC(hipCtxSetCurrent, original_); }
} catch (const std::runtime_error &e) {
error::wrap_c_api(status::runtime_error, e.what());
}
}

#pragma clang diagnostic pop

} 
} 
} 
} 
