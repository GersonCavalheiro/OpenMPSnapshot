




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/system/error_code.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_driver_types.h>

namespace hydra_thrust
{

namespace system
{

namespace cuda
{



namespace errc
{


enum errc_t
{
success                            = cudaSuccess,
missing_configuration              = cudaErrorMissingConfiguration,
memory_allocation                  = cudaErrorMemoryAllocation,
initialization_error               = cudaErrorInitializationError,
launch_failure                     = cudaErrorLaunchFailure,
prior_launch_failure               = cudaErrorPriorLaunchFailure,
launch_timeout                     = cudaErrorLaunchTimeout,
launch_out_of_resources            = cudaErrorLaunchOutOfResources,
invalid_device_function            = cudaErrorInvalidDeviceFunction,
invalid_configuration              = cudaErrorInvalidConfiguration,
invalid_device                     = cudaErrorInvalidDevice,
invalid_value                      = cudaErrorInvalidValue,
invalid_pitch_value                = cudaErrorInvalidPitchValue,
invalid_symbol                     = cudaErrorInvalidSymbol,
map_buffer_object_failed           = cudaErrorMapBufferObjectFailed,
unmap_buffer_object_failed         = cudaErrorUnmapBufferObjectFailed,
invalid_host_pointer               = cudaErrorInvalidHostPointer,
invalid_device_pointer             = cudaErrorInvalidDevicePointer,
invalid_texture                    = cudaErrorInvalidTexture,
invalid_texture_binding            = cudaErrorInvalidTextureBinding,
invalid_channel_descriptor         = cudaErrorInvalidChannelDescriptor,
invalid_memcpy_direction           = cudaErrorInvalidMemcpyDirection,
address_of_constant_error          = cudaErrorAddressOfConstant,
texture_fetch_failed               = cudaErrorTextureFetchFailed,
texture_not_bound                  = cudaErrorTextureNotBound,
synchronization_error              = cudaErrorSynchronizationError,
invalid_filter_setting             = cudaErrorInvalidFilterSetting,
invalid_norm_setting               = cudaErrorInvalidNormSetting,
mixed_device_execution             = cudaErrorMixedDeviceExecution,
cuda_runtime_unloading             = cudaErrorCudartUnloading,
unknown                            = cudaErrorUnknown,
not_yet_implemented                = cudaErrorNotYetImplemented,
memory_value_too_large             = cudaErrorMemoryValueTooLarge,
invalid_resource_handle            = cudaErrorInvalidResourceHandle,
not_ready                          = cudaErrorNotReady,
insufficient_driver                = cudaErrorInsufficientDriver,
set_on_active_process_error        = cudaErrorSetOnActiveProcess,
no_device                          = cudaErrorNoDevice,
ecc_uncorrectable                  = cudaErrorECCUncorrectable,

#if CUDART_VERSION >= 4020
shared_object_symbol_not_found     = cudaErrorSharedObjectSymbolNotFound,
shared_object_init_failed          = cudaErrorSharedObjectInitFailed,
unsupported_limit                  = cudaErrorUnsupportedLimit,
duplicate_variable_name            = cudaErrorDuplicateVariableName,
duplicate_texture_name             = cudaErrorDuplicateTextureName,
duplicate_surface_name             = cudaErrorDuplicateSurfaceName,
devices_unavailable                = cudaErrorDevicesUnavailable,
invalid_kernel_image               = cudaErrorInvalidKernelImage,
no_kernel_image_for_device         = cudaErrorNoKernelImageForDevice,
incompatible_driver_context        = cudaErrorIncompatibleDriverContext,
peer_access_already_enabled        = cudaErrorPeerAccessAlreadyEnabled,
peer_access_not_enabled            = cudaErrorPeerAccessNotEnabled,
device_already_in_use              = cudaErrorDeviceAlreadyInUse,
profiler_disabled                  = cudaErrorProfilerDisabled,
assert_triggered                   = cudaErrorAssert,
too_many_peers                     = cudaErrorTooManyPeers,
host_memory_already_registered     = cudaErrorHostMemoryAlreadyRegistered,
host_memory_not_registered         = cudaErrorHostMemoryNotRegistered,
operating_system_error             = cudaErrorOperatingSystem,
#endif

#if CUDART_VERSION >= 5000
peer_access_unsupported            = cudaErrorPeerAccessUnsupported,
launch_max_depth_exceeded          = cudaErrorLaunchMaxDepthExceeded,
launch_file_scoped_texture_used    = cudaErrorLaunchFileScopedTex,
launch_file_scoped_surface_used    = cudaErrorLaunchFileScopedSurf,
sync_depth_exceeded                = cudaErrorSyncDepthExceeded,
attempted_operation_not_permitted  = cudaErrorNotPermitted,
attempted_operation_not_supported  = cudaErrorNotSupported,
#endif

startup_failure                    = cudaErrorStartupFailure
}; 


} 

} 


inline const error_category &cuda_category(void);




template<> struct is_error_code_enum<cuda::errc::errc_t> : hydra_thrust::detail::true_type {};



inline error_code make_error_code(cuda::errc::errc_t e);



inline error_condition make_error_condition(cuda::errc::errc_t e);

} 

namespace cuda_cub
{
namespace errc = system::cuda::errc;
} 

namespace cuda
{
namespace errc = system::cuda::errc;
} 

using system::cuda_category;

} 

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/error.inl>

