
#ifndef _ZE_API_H
#define _ZE_API_H
#if defined(__cplusplus)
#pragma once
#endif

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(__GNUC__)
#pragma region common
#endif
#ifndef ZE_MAKE_VERSION
#define ZE_MAKE_VERSION( _major, _minor )  (( _major << 16 )|( _minor & 0x0000ffff))
#endif 

#ifndef ZE_MAJOR_VERSION
#define ZE_MAJOR_VERSION( _ver )  ( _ver >> 16 )
#endif 

#ifndef ZE_MINOR_VERSION
#define ZE_MINOR_VERSION( _ver )  ( _ver & 0x0000ffff )
#endif 

#ifndef ZE_APICALL
#if defined(_WIN32)
#define ZE_APICALL  __cdecl
#else
#define ZE_APICALL  
#endif 
#endif 

#ifndef ZE_APIEXPORT
#if defined(_WIN32)
#define ZE_APIEXPORT  __declspec(dllexport)
#endif 
#endif 

#ifndef ZE_APIEXPORT
#if __GNUC__ >= 4
#define ZE_APIEXPORT  __attribute__ ((visibility ("default")))
#else
#define ZE_APIEXPORT  
#endif 
#endif 

#ifndef ZE_DLLEXPORT
#if defined(_WIN32)
#define ZE_DLLEXPORT  __declspec(dllexport)
#endif 
#endif 

#ifndef ZE_DLLEXPORT
#if __GNUC__ >= 4
#define ZE_DLLEXPORT  __attribute__ ((visibility ("default")))
#else
#define ZE_DLLEXPORT  
#endif 
#endif 

typedef uint8_t ze_bool_t;

typedef struct _ze_driver_handle_t *ze_driver_handle_t;

typedef struct _ze_device_handle_t *ze_device_handle_t;

typedef struct _ze_context_handle_t *ze_context_handle_t;

typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;

typedef struct _ze_command_list_handle_t *ze_command_list_handle_t;

typedef struct _ze_fence_handle_t *ze_fence_handle_t;

typedef struct _ze_event_pool_handle_t *ze_event_pool_handle_t;

typedef struct _ze_event_handle_t *ze_event_handle_t;

typedef struct _ze_image_handle_t *ze_image_handle_t;

typedef struct _ze_module_handle_t *ze_module_handle_t;

typedef struct _ze_module_build_log_handle_t *ze_module_build_log_handle_t;

typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;

typedef struct _ze_sampler_handle_t *ze_sampler_handle_t;

typedef struct _ze_physical_mem_handle_t *ze_physical_mem_handle_t;

#ifndef ZE_MAX_IPC_HANDLE_SIZE
#define ZE_MAX_IPC_HANDLE_SIZE  64
#endif 

typedef struct _ze_ipc_mem_handle_t
{
char data[ZE_MAX_IPC_HANDLE_SIZE];              

} ze_ipc_mem_handle_t;

typedef struct _ze_ipc_event_pool_handle_t
{
char data[ZE_MAX_IPC_HANDLE_SIZE];              

} ze_ipc_event_pool_handle_t;

#ifndef ZE_BIT
#define ZE_BIT( _i )  ( 1 << _i )
#endif 

typedef enum _ze_result_t
{
ZE_RESULT_SUCCESS = 0,                          
ZE_RESULT_NOT_READY = 1,                        
ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001,       
ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 0x70000002,
ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003,  
ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 0x70000004,  
ZE_RESULT_ERROR_MODULE_LINK_FAILURE = 0x70000005,   
ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 0x70000006, 
ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 0x70000007, 
ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000,  
ZE_RESULT_ERROR_NOT_AVAILABLE = 0x70010001,     
ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 0x70020000,
ZE_RESULT_ERROR_UNINITIALIZED = 0x78000001,     
ZE_RESULT_ERROR_UNSUPPORTED_VERSION = 0x78000002,   
ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 0x78000003,   
ZE_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,  
ZE_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,   
ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 0x78000006,  
ZE_RESULT_ERROR_INVALID_NULL_POINTER = 0x78000007,  
ZE_RESULT_ERROR_INVALID_SIZE = 0x78000008,      
ZE_RESULT_ERROR_UNSUPPORTED_SIZE = 0x78000009,  
ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 0x7800000a, 
ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x7800000b,
ZE_RESULT_ERROR_INVALID_ENUMERATION = 0x7800000c,   
ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 0x7800000d,   
ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x7800000e,  
ZE_RESULT_ERROR_INVALID_NATIVE_BINARY = 0x7800000f, 
ZE_RESULT_ERROR_INVALID_GLOBAL_NAME = 0x78000010,   
ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 0x78000011,   
ZE_RESULT_ERROR_INVALID_FUNCTION_NAME = 0x78000012, 
ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x78000013,  
ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x78000014,
ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x78000015, 
ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x78000016,  
ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x78000017,
ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 0x78000018,   
ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000019, 
ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 0x7800001a,   
ZE_RESULT_ERROR_UNKNOWN = 0x7ffffffe,           
ZE_RESULT_FORCE_UINT32 = 0x7fffffff

} ze_result_t;

typedef enum _ze_structure_type_t
{
ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1,      
ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES = 0x2,  
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3,      
ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x4,  
ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 0x5,   
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6, 
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 0x7,   
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES = 0x8,
ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES = 0x9,
ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES = 0xa,
ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 0xb,  
ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xc,  
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xd,           
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xe,     
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0xf,      
ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 0x10,       
ZE_STRUCTURE_TYPE_EVENT_DESC = 0x11,            
ZE_STRUCTURE_TYPE_FENCE_DESC = 0x12,            
ZE_STRUCTURE_TYPE_IMAGE_DESC = 0x13,            
ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES = 0x14,      
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15, 
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16,   
ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x17,  
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18,   
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 0x19, 
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 0x1a, 
ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b,           
ZE_STRUCTURE_TYPE_MODULE_PROPERTIES = 0x1c,     
ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d,           
ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES = 0x1e,     
ZE_STRUCTURE_TYPE_SAMPLER_DESC = 0x1f,          
ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 0x20,     
ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 0x21,
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32 = 0x22,  
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32 = 0x23,  
ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES = 0x00010001,
ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC = 0x10002,  
ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES = 0x10003,
ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC = 0x10004, 
ZE_STRUCTURE_TYPE_EU_COUNT_EXT = 0x10005,       
ZE_STRUCTURE_TYPE_SRGB_EXT_DESC = 0x10006,      
ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC = 0x10007,
ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 0x10008, 
ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES = 0x10009,  
ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC = 0x1000a,   
ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC = 0x1000b,  
ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES = 0x1000c,
ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 0x00020001,  
ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC = 0x00020002, 
ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES = 0x00020003,  
ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC = 0x00020004,
ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC = 0x00020005,  
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 = 0x00020006,   
ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES = 0x00020007, 
ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC = 0x00020008,  
ZE_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_structure_type_t;

typedef uint32_t ze_external_memory_type_flags_t;
typedef enum _ze_external_memory_type_flag_t
{
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = ZE_BIT(0), 
ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = ZE_BIT(1),   
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 = ZE_BIT(2),  
ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT = ZE_BIT(3),  
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE = ZE_BIT(4), 
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT = ZE_BIT(5), 
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP = ZE_BIT(6),
ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE = ZE_BIT(7),
ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_external_memory_type_flag_t;

typedef struct _ze_base_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    

} ze_base_properties_t;

typedef struct _ze_base_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              

} ze_base_desc_t;




typedef struct _ze_ipc_mem_handle_t ze_ipc_mem_handle_t;

typedef struct _ze_ipc_event_pool_handle_t ze_ipc_event_pool_handle_t;

typedef struct _ze_base_properties_t ze_base_properties_t;

typedef struct _ze_base_desc_t ze_base_desc_t;

typedef struct _ze_driver_uuid_t ze_driver_uuid_t;

typedef struct _ze_driver_properties_t ze_driver_properties_t;

typedef struct _ze_driver_ipc_properties_t ze_driver_ipc_properties_t;

typedef struct _ze_driver_extension_properties_t ze_driver_extension_properties_t;

typedef struct _ze_device_uuid_t ze_device_uuid_t;

typedef struct _ze_device_properties_t ze_device_properties_t;

typedef struct _ze_device_thread_t ze_device_thread_t;

typedef struct _ze_device_compute_properties_t ze_device_compute_properties_t;

typedef struct _ze_native_kernel_uuid_t ze_native_kernel_uuid_t;

typedef struct _ze_device_module_properties_t ze_device_module_properties_t;

typedef struct _ze_command_queue_group_properties_t ze_command_queue_group_properties_t;

typedef struct _ze_device_memory_properties_t ze_device_memory_properties_t;

typedef struct _ze_device_memory_access_properties_t ze_device_memory_access_properties_t;

typedef struct _ze_device_cache_properties_t ze_device_cache_properties_t;

typedef struct _ze_device_image_properties_t ze_device_image_properties_t;

typedef struct _ze_device_external_memory_properties_t ze_device_external_memory_properties_t;

typedef struct _ze_device_p2p_properties_t ze_device_p2p_properties_t;

typedef struct _ze_context_desc_t ze_context_desc_t;

typedef struct _ze_command_queue_desc_t ze_command_queue_desc_t;

typedef struct _ze_command_list_desc_t ze_command_list_desc_t;

typedef struct _ze_copy_region_t ze_copy_region_t;

typedef struct _ze_image_region_t ze_image_region_t;

typedef struct _ze_event_pool_desc_t ze_event_pool_desc_t;

typedef struct _ze_event_desc_t ze_event_desc_t;

typedef struct _ze_kernel_timestamp_data_t ze_kernel_timestamp_data_t;

typedef struct _ze_kernel_timestamp_result_t ze_kernel_timestamp_result_t;

typedef struct _ze_fence_desc_t ze_fence_desc_t;

typedef struct _ze_image_format_t ze_image_format_t;

typedef struct _ze_image_desc_t ze_image_desc_t;

typedef struct _ze_image_properties_t ze_image_properties_t;

typedef struct _ze_device_mem_alloc_desc_t ze_device_mem_alloc_desc_t;

typedef struct _ze_host_mem_alloc_desc_t ze_host_mem_alloc_desc_t;

typedef struct _ze_memory_allocation_properties_t ze_memory_allocation_properties_t;

typedef struct _ze_external_memory_export_desc_t ze_external_memory_export_desc_t;

typedef struct _ze_external_memory_import_fd_t ze_external_memory_import_fd_t;

typedef struct _ze_external_memory_export_fd_t ze_external_memory_export_fd_t;

typedef struct _ze_external_memory_import_win32_handle_t ze_external_memory_import_win32_handle_t;

typedef struct _ze_external_memory_export_win32_handle_t ze_external_memory_export_win32_handle_t;

typedef struct _ze_module_constants_t ze_module_constants_t;

typedef struct _ze_module_desc_t ze_module_desc_t;

typedef struct _ze_module_properties_t ze_module_properties_t;

typedef struct _ze_kernel_desc_t ze_kernel_desc_t;

typedef struct _ze_kernel_uuid_t ze_kernel_uuid_t;

typedef struct _ze_kernel_properties_t ze_kernel_properties_t;

typedef struct _ze_kernel_preferred_group_size_properties_t ze_kernel_preferred_group_size_properties_t;

typedef struct _ze_group_count_t ze_group_count_t;

typedef struct _ze_module_program_exp_desc_t ze_module_program_exp_desc_t;

typedef struct _ze_device_raytracing_ext_properties_t ze_device_raytracing_ext_properties_t;

typedef struct _ze_raytracing_mem_alloc_ext_desc_t ze_raytracing_mem_alloc_ext_desc_t;

typedef struct _ze_sampler_desc_t ze_sampler_desc_t;

typedef struct _ze_physical_mem_desc_t ze_physical_mem_desc_t;

typedef struct _ze_float_atomic_ext_properties_t ze_float_atomic_ext_properties_t;

typedef struct _ze_relaxed_allocation_limits_exp_desc_t ze_relaxed_allocation_limits_exp_desc_t;

typedef struct _ze_cache_reservation_ext_desc_t ze_cache_reservation_ext_desc_t;

typedef struct _ze_image_memory_properties_exp_t ze_image_memory_properties_exp_t;

typedef struct _ze_image_view_planar_exp_desc_t ze_image_view_planar_exp_desc_t;

typedef struct _ze_scheduling_hint_exp_properties_t ze_scheduling_hint_exp_properties_t;

typedef struct _ze_scheduling_hint_exp_desc_t ze_scheduling_hint_exp_desc_t;

typedef struct _ze_context_power_saving_hint_exp_desc_t ze_context_power_saving_hint_exp_desc_t;

typedef struct _ze_eu_count_ext_t ze_eu_count_ext_t;

typedef struct _ze_pci_address_ext_t ze_pci_address_ext_t;

typedef struct _ze_pci_speed_ext_t ze_pci_speed_ext_t;

typedef struct _ze_pci_ext_properties_t ze_pci_ext_properties_t;

typedef struct _ze_srgb_ext_desc_t ze_srgb_ext_desc_t;

typedef struct _ze_image_allocation_ext_properties_t ze_image_allocation_ext_properties_t;

typedef struct _ze_linkage_inspection_ext_desc_t ze_linkage_inspection_ext_desc_t;

typedef struct _ze_memory_compression_hints_ext_desc_t ze_memory_compression_hints_ext_desc_t;

typedef struct _ze_driver_memory_free_ext_properties_t ze_driver_memory_free_ext_properties_t;

typedef struct _ze_memory_free_ext_desc_t ze_memory_free_ext_desc_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region driver
#endif
typedef uint32_t ze_init_flags_t;
typedef enum _ze_init_flag_t
{
ZE_INIT_FLAG_GPU_ONLY = ZE_BIT(0),              
ZE_INIT_FLAG_VPU_ONLY = ZE_BIT(1),              
ZE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_init_flag_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeInit(
ze_init_flags_t flags                           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGet(
uint32_t* pCount,                               
ze_driver_handle_t* phDrivers                   
);

typedef enum _ze_api_version_t
{
ZE_API_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),   
ZE_API_VERSION_1_1 = ZE_MAKE_VERSION( 1, 1 ),   
ZE_API_VERSION_1_2 = ZE_MAKE_VERSION( 1, 2 ),   
ZE_API_VERSION_1_3 = ZE_MAKE_VERSION( 1, 3 ),   
ZE_API_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 3 ),   
ZE_API_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_api_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetApiVersion(
ze_driver_handle_t hDriver,                     
ze_api_version_t* version                       
);

#ifndef ZE_MAX_DRIVER_UUID_SIZE
#define ZE_MAX_DRIVER_UUID_SIZE  16
#endif 

typedef struct _ze_driver_uuid_t
{
uint8_t id[ZE_MAX_DRIVER_UUID_SIZE];            

} ze_driver_uuid_t;

typedef struct _ze_driver_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_driver_uuid_t uuid;                          
uint32_t driverVersion;                         

} ze_driver_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetProperties(
ze_driver_handle_t hDriver,                     
ze_driver_properties_t* pDriverProperties       
);

typedef uint32_t ze_ipc_property_flags_t;
typedef enum _ze_ipc_property_flag_t
{
ZE_IPC_PROPERTY_FLAG_MEMORY = ZE_BIT(0),        
ZE_IPC_PROPERTY_FLAG_EVENT_POOL = ZE_BIT(1),    
ZE_IPC_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_ipc_property_flag_t;

typedef struct _ze_driver_ipc_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_ipc_property_flags_t flags;                  

} ze_driver_ipc_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetIpcProperties(
ze_driver_handle_t hDriver,                     
ze_driver_ipc_properties_t* pIpcProperties      
);

#ifndef ZE_MAX_EXTENSION_NAME
#define ZE_MAX_EXTENSION_NAME  256
#endif 

typedef struct _ze_driver_extension_properties_t
{
char name[ZE_MAX_EXTENSION_NAME];               
uint32_t version;                               

} ze_driver_extension_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetExtensionProperties(
ze_driver_handle_t hDriver,                     
uint32_t* pCount,                               
ze_driver_extension_properties_t* pExtensionProperties  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDriverGetExtensionFunctionAddress(
ze_driver_handle_t hDriver,                     
const char* name,                               
void** ppFunctionAddress                        
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region device
#endif
ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGet(
ze_driver_handle_t hDriver,                     
uint32_t* pCount,                               
ze_device_handle_t* phDevices                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetSubDevices(
ze_device_handle_t hDevice,                     
uint32_t* pCount,                               
ze_device_handle_t* phSubdevices                
);

typedef enum _ze_device_type_t
{
ZE_DEVICE_TYPE_GPU = 1,                         
ZE_DEVICE_TYPE_CPU = 2,                         
ZE_DEVICE_TYPE_FPGA = 3,                        
ZE_DEVICE_TYPE_MCA = 4,                         
ZE_DEVICE_TYPE_VPU = 5,                         
ZE_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_device_type_t;

#ifndef ZE_MAX_DEVICE_UUID_SIZE
#define ZE_MAX_DEVICE_UUID_SIZE  16
#endif 

typedef struct _ze_device_uuid_t
{
uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];            

} ze_device_uuid_t;

#ifndef ZE_MAX_DEVICE_NAME
#define ZE_MAX_DEVICE_NAME  256
#endif 

typedef uint32_t ze_device_property_flags_t;
typedef enum _ze_device_property_flag_t
{
ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0), 
ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = ZE_BIT(1),  
ZE_DEVICE_PROPERTY_FLAG_ECC = ZE_BIT(2),        
ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = ZE_BIT(3), 
ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_property_flag_t;

typedef struct _ze_device_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_type_t type;                          
uint32_t vendorId;                              
uint32_t deviceId;                              
ze_device_property_flags_t flags;               
uint32_t subdeviceId;                           
uint32_t coreClockRate;                         
uint64_t maxMemAllocSize;                       
uint32_t maxHardwareContexts;                   
uint32_t maxCommandQueuePriority;               
uint32_t numThreadsPerEU;                       
uint32_t physicalEUSimdWidth;                   
uint32_t numEUsPerSubslice;                     
uint32_t numSubslicesPerSlice;                  
uint32_t numSlices;                             
uint64_t timerResolution;                       
uint32_t timestampValidBits;                    
uint32_t kernelTimestampValidBits;              
ze_device_uuid_t uuid;                          
char name[ZE_MAX_DEVICE_NAME];                  

} ze_device_properties_t;

typedef struct _ze_device_thread_t
{
uint32_t slice;                                 
uint32_t subslice;                              
uint32_t eu;                                    
uint32_t thread;                                

} ze_device_thread_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetProperties(
ze_device_handle_t hDevice,                     
ze_device_properties_t* pDeviceProperties       
);

#ifndef ZE_SUBGROUPSIZE_COUNT
#define ZE_SUBGROUPSIZE_COUNT  8
#endif 

typedef struct _ze_device_compute_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint32_t maxTotalGroupSize;                     
uint32_t maxGroupSizeX;                         
uint32_t maxGroupSizeY;                         
uint32_t maxGroupSizeZ;                         
uint32_t maxGroupCountX;                        
uint32_t maxGroupCountY;                        
uint32_t maxGroupCountZ;                        
uint32_t maxSharedLocalMemory;                  
uint32_t numSubGroupSizes;                      
uint32_t subGroupSizes[ZE_SUBGROUPSIZE_COUNT];  

} ze_device_compute_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetComputeProperties(
ze_device_handle_t hDevice,                     
ze_device_compute_properties_t* pComputeProperties  
);

#ifndef ZE_MAX_NATIVE_KERNEL_UUID_SIZE
#define ZE_MAX_NATIVE_KERNEL_UUID_SIZE  16
#endif 

typedef struct _ze_native_kernel_uuid_t
{
uint8_t id[ZE_MAX_NATIVE_KERNEL_UUID_SIZE];     

} ze_native_kernel_uuid_t;

typedef uint32_t ze_device_module_flags_t;
typedef enum _ze_device_module_flag_t
{
ZE_DEVICE_MODULE_FLAG_FP16 = ZE_BIT(0),         
ZE_DEVICE_MODULE_FLAG_FP64 = ZE_BIT(1),         
ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = ZE_BIT(2),
ZE_DEVICE_MODULE_FLAG_DP4A = ZE_BIT(3),         
ZE_DEVICE_MODULE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_module_flag_t;

typedef uint32_t ze_device_fp_flags_t;
typedef enum _ze_device_fp_flag_t
{
ZE_DEVICE_FP_FLAG_DENORM = ZE_BIT(0),           
ZE_DEVICE_FP_FLAG_INF_NAN = ZE_BIT(1),          
ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST = ZE_BIT(2), 
ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO = ZE_BIT(3),    
ZE_DEVICE_FP_FLAG_ROUND_TO_INF = ZE_BIT(4),     
ZE_DEVICE_FP_FLAG_FMA = ZE_BIT(5),              
ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT = ZE_BIT(6),  
ZE_DEVICE_FP_FLAG_SOFT_FLOAT = ZE_BIT(7),       
ZE_DEVICE_FP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_fp_flag_t;

typedef struct _ze_device_module_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint32_t spirvVersionSupported;                 
ze_device_module_flags_t flags;                 
ze_device_fp_flags_t fp16flags;                 
ze_device_fp_flags_t fp32flags;                 
ze_device_fp_flags_t fp64flags;                 
uint32_t maxArgumentsSize;                      
uint32_t printfBufferSize;                      
ze_native_kernel_uuid_t nativeKernelSupported;  

} ze_device_module_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetModuleProperties(
ze_device_handle_t hDevice,                     
ze_device_module_properties_t* pModuleProperties
);

typedef uint32_t ze_command_queue_group_property_flags_t;
typedef enum _ze_command_queue_group_property_flag_t
{
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = ZE_BIT(0),   
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = ZE_BIT(1),  
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = ZE_BIT(2),   
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = ZE_BIT(3),   
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_group_property_flag_t;

typedef struct _ze_command_queue_group_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_command_queue_group_property_flags_t flags;  
size_t maxMemoryFillPatternSize;                
uint32_t numQueues;                             

} ze_command_queue_group_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetCommandQueueGroupProperties(
ze_device_handle_t hDevice,                     
uint32_t* pCount,                               
ze_command_queue_group_properties_t* pCommandQueueGroupProperties   
);

typedef uint32_t ze_device_memory_property_flags_t;
typedef enum _ze_device_memory_property_flag_t
{
ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD = ZE_BIT(0), 
ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_memory_property_flag_t;

typedef struct _ze_device_memory_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_memory_property_flags_t flags;        
uint32_t maxClockRate;                          
uint32_t maxBusWidth;                           
uint64_t totalSize;                             
char name[ZE_MAX_DEVICE_NAME];                  

} ze_device_memory_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetMemoryProperties(
ze_device_handle_t hDevice,                     
uint32_t* pCount,                               
ze_device_memory_properties_t* pMemProperties   
);

typedef uint32_t ze_memory_access_cap_flags_t;
typedef enum _ze_memory_access_cap_flag_t
{
ZE_MEMORY_ACCESS_CAP_FLAG_RW = ZE_BIT(0),       
ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC = ZE_BIT(1),   
ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT = ZE_BIT(2),   
ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC = ZE_BIT(3),
ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_memory_access_cap_flag_t;

typedef struct _ze_device_memory_access_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_memory_access_cap_flags_t hostAllocCapabilities; 
ze_memory_access_cap_flags_t deviceAllocCapabilities;   
ze_memory_access_cap_flags_t sharedSingleDeviceAllocCapabilities;   
ze_memory_access_cap_flags_t sharedCrossDeviceAllocCapabilities;
ze_memory_access_cap_flags_t sharedSystemAllocCapabilities; 

} ze_device_memory_access_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetMemoryAccessProperties(
ze_device_handle_t hDevice,                     
ze_device_memory_access_properties_t* pMemAccessProperties  
);

typedef uint32_t ze_device_cache_property_flags_t;
typedef enum _ze_device_cache_property_flag_t
{
ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL = ZE_BIT(0), 
ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_cache_property_flag_t;

typedef struct _ze_device_cache_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_cache_property_flags_t flags;         
size_t cacheSize;                               

} ze_device_cache_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetCacheProperties(
ze_device_handle_t hDevice,                     
uint32_t* pCount,                               
ze_device_cache_properties_t* pCacheProperties  
);

typedef struct _ze_device_image_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint32_t maxImageDims1D;                        
uint32_t maxImageDims2D;                        
uint32_t maxImageDims3D;                        
uint64_t maxImageBufferSize;                    
uint32_t maxImageArraySlices;                   
uint32_t maxSamplers;                           
uint32_t maxReadImageArgs;                      
uint32_t maxWriteImageArgs;                     

} ze_device_image_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetImageProperties(
ze_device_handle_t hDevice,                     
ze_device_image_properties_t* pImageProperties  
);

typedef struct _ze_device_external_memory_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_external_memory_type_flags_t memoryAllocationImportTypes;
ze_external_memory_type_flags_t memoryAllocationExportTypes;
ze_external_memory_type_flags_t imageImportTypes;   
ze_external_memory_type_flags_t imageExportTypes;   

} ze_device_external_memory_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetExternalMemoryProperties(
ze_device_handle_t hDevice,                     
ze_device_external_memory_properties_t* pExternalMemoryProperties   
);

typedef uint32_t ze_device_p2p_property_flags_t;
typedef enum _ze_device_p2p_property_flag_t
{
ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = ZE_BIT(0), 
ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = ZE_BIT(1),
ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_p2p_property_flag_t;

typedef struct _ze_device_p2p_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_p2p_property_flags_t flags;           

} ze_device_p2p_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetP2PProperties(
ze_device_handle_t hDevice,                     
ze_device_handle_t hPeerDevice,                 
ze_device_p2p_properties_t* pP2PProperties      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceCanAccessPeer(
ze_device_handle_t hDevice,                     
ze_device_handle_t hPeerDevice,                 
ze_bool_t* value                                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetStatus(
ze_device_handle_t hDevice                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceGetGlobalTimestamps(
ze_device_handle_t hDevice,                     
uint64_t* hostTimestamp,                        
uint64_t* deviceTimestamp                       
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region context
#endif
typedef uint32_t ze_context_flags_t;
typedef enum _ze_context_flag_t
{
ZE_CONTEXT_FLAG_TBD = ZE_BIT(0),                
ZE_CONTEXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_context_flag_t;

typedef struct _ze_context_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_context_flags_t flags;                       

} ze_context_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextCreate(
ze_driver_handle_t hDriver,                     
const ze_context_desc_t* desc,                  
ze_context_handle_t* phContext                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextCreateEx(
ze_driver_handle_t hDriver,                     
const ze_context_desc_t* desc,                  
uint32_t numDevices,                            
ze_device_handle_t* phDevices,                  
ze_context_handle_t* phContext                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextDestroy(
ze_context_handle_t hContext                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextGetStatus(
ze_context_handle_t hContext                    
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region cmdqueue
#endif
typedef uint32_t ze_command_queue_flags_t;
typedef enum _ze_command_queue_flag_t
{
ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = ZE_BIT(0),
ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_flag_t;

typedef enum _ze_command_queue_mode_t
{
ZE_COMMAND_QUEUE_MODE_DEFAULT = 0,              
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1,          
ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2,         
ZE_COMMAND_QUEUE_MODE_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_mode_t;

typedef enum _ze_command_queue_priority_t
{
ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,           
ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1,     
ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2,    
ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32 = 0x7fffffff

} ze_command_queue_priority_t;

typedef struct _ze_command_queue_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t ordinal;                               
uint32_t index;                                 
ze_command_queue_flags_t flags;                 
ze_command_queue_mode_t mode;                   
ze_command_queue_priority_t priority;           

} ze_command_queue_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_command_queue_desc_t* desc,            
ze_command_queue_handle_t* phCommandQueue       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueDestroy(
ze_command_queue_handle_t hCommandQueue         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueExecuteCommandLists(
ze_command_queue_handle_t hCommandQueue,        
uint32_t numCommandLists,                       
ze_command_list_handle_t* phCommandLists,       
ze_fence_handle_t hFence                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandQueueSynchronize(
ze_command_queue_handle_t hCommandQueue,        
uint64_t timeout                                
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region cmdlist
#endif
typedef uint32_t ze_command_list_flags_t;
typedef enum _ze_command_list_flag_t
{
ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = ZE_BIT(0),  
ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = ZE_BIT(1),   
ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = ZE_BIT(2), 
ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_command_list_flag_t;

typedef struct _ze_command_list_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t commandQueueGroupOrdinal;              
ze_command_list_flags_t flags;                  

} ze_command_list_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_command_list_desc_t* desc,             
ze_command_list_handle_t* phCommandList         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListCreateImmediate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_command_queue_desc_t* altdesc,         
ze_command_list_handle_t* phCommandList         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListDestroy(
ze_command_list_handle_t hCommandList           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListClose(
ze_command_list_handle_t hCommandList           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListReset(
ze_command_list_handle_t hCommandList           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendWriteGlobalTimestamp(
ze_command_list_handle_t hCommandList,          
uint64_t* dstptr,                               
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region barrier
#endif
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendBarrier(
ze_command_list_handle_t hCommandList,          
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryRangesBarrier(
ze_command_list_handle_t hCommandList,          
uint32_t numRanges,                             
const size_t* pRangeSizes,                      
const void** pRanges,                           
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextSystemBarrier(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice                      
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region copy
#endif
ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopy(
ze_command_list_handle_t hCommandList,          
void* dstptr,                                   
const void* srcptr,                             
size_t size,                                    
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryFill(
ze_command_list_handle_t hCommandList,          
void* ptr,                                      
const void* pattern,                            
size_t pattern_size,                            
size_t size,                                    
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

typedef struct _ze_copy_region_t
{
uint32_t originX;                               
uint32_t originY;                               
uint32_t originZ;                               
uint32_t width;                                 
uint32_t height;                                
uint32_t depth;                                 

} ze_copy_region_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopyRegion(
ze_command_list_handle_t hCommandList,          
void* dstptr,                                   
const ze_copy_region_t* dstRegion,              
uint32_t dstPitch,                              
uint32_t dstSlicePitch,                         
const void* srcptr,                             
const ze_copy_region_t* srcRegion,              
uint32_t srcPitch,                              
uint32_t srcSlicePitch,                         
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryCopyFromContext(
ze_command_list_handle_t hCommandList,          
void* dstptr,                                   
ze_context_handle_t hContextSrc,                
const void* srcptr,                             
size_t size,                                    
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopy(
ze_command_list_handle_t hCommandList,          
ze_image_handle_t hDstImage,                    
ze_image_handle_t hSrcImage,                    
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

typedef struct _ze_image_region_t
{
uint32_t originX;                               
uint32_t originY;                               
uint32_t originZ;                               
uint32_t width;                                 
uint32_t height;                                
uint32_t depth;                                 

} ze_image_region_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyRegion(
ze_command_list_handle_t hCommandList,          
ze_image_handle_t hDstImage,                    
ze_image_handle_t hSrcImage,                    
const ze_image_region_t* pDstRegion,            
const ze_image_region_t* pSrcRegion,            
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyToMemory(
ze_command_list_handle_t hCommandList,          
void* dstptr,                                   
ze_image_handle_t hSrcImage,                    
const ze_image_region_t* pSrcRegion,            
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyFromMemory(
ze_command_list_handle_t hCommandList,          
ze_image_handle_t hDstImage,                    
const void* srcptr,                             
const ze_image_region_t* pDstRegion,            
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemoryPrefetch(
ze_command_list_handle_t hCommandList,          
const void* ptr,                                
size_t size                                     
);

typedef enum _ze_memory_advice_t
{
ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0,           
ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1,         
ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2,    
ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3,  
ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 4,     
ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 5,   
ZE_MEMORY_ADVICE_BIAS_CACHED = 6,               
ZE_MEMORY_ADVICE_BIAS_UNCACHED = 7,             
ZE_MEMORY_ADVICE_FORCE_UINT32 = 0x7fffffff

} ze_memory_advice_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendMemAdvise(
ze_command_list_handle_t hCommandList,          
ze_device_handle_t hDevice,                     
const void* ptr,                                
size_t size,                                    
ze_memory_advice_t advice                       
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region event
#endif
typedef uint32_t ze_event_pool_flags_t;
typedef enum _ze_event_pool_flag_t
{
ZE_EVENT_POOL_FLAG_HOST_VISIBLE = ZE_BIT(0),    
ZE_EVENT_POOL_FLAG_IPC = ZE_BIT(1),             
ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = ZE_BIT(2),
ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_event_pool_flag_t;

typedef struct _ze_event_pool_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_event_pool_flags_t flags;                    
uint32_t count;                                 

} ze_event_pool_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolCreate(
ze_context_handle_t hContext,                   
const ze_event_pool_desc_t* desc,               
uint32_t numDevices,                            
ze_device_handle_t* phDevices,                  
ze_event_pool_handle_t* phEventPool             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolDestroy(
ze_event_pool_handle_t hEventPool               
);

typedef uint32_t ze_event_scope_flags_t;
typedef enum _ze_event_scope_flag_t
{
ZE_EVENT_SCOPE_FLAG_SUBDEVICE = ZE_BIT(0),      
ZE_EVENT_SCOPE_FLAG_DEVICE = ZE_BIT(1),         
ZE_EVENT_SCOPE_FLAG_HOST = ZE_BIT(2),           
ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_event_scope_flag_t;

typedef struct _ze_event_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t index;                                 
ze_event_scope_flags_t signal;                  
ze_event_scope_flags_t wait;                    

} ze_event_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventCreate(
ze_event_pool_handle_t hEventPool,              
const ze_event_desc_t* desc,                    
ze_event_handle_t* phEvent                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventDestroy(
ze_event_handle_t hEvent                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolGetIpcHandle(
ze_event_pool_handle_t hEventPool,              
ze_ipc_event_pool_handle_t* phIpc               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolOpenIpcHandle(
ze_context_handle_t hContext,                   
ze_ipc_event_pool_handle_t hIpc,                
ze_event_pool_handle_t* phEventPool             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventPoolCloseIpcHandle(
ze_event_pool_handle_t hEventPool               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendSignalEvent(
ze_command_list_handle_t hCommandList,          
ze_event_handle_t hEvent                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendWaitOnEvents(
ze_command_list_handle_t hCommandList,          
uint32_t numEvents,                             
ze_event_handle_t* phEvents                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostSignal(
ze_event_handle_t hEvent                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostSynchronize(
ze_event_handle_t hEvent,                       
uint64_t timeout                                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryStatus(
ze_event_handle_t hEvent                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendEventReset(
ze_command_list_handle_t hCommandList,          
ze_event_handle_t hEvent                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventHostReset(
ze_event_handle_t hEvent                        
);

typedef struct _ze_kernel_timestamp_data_t
{
uint64_t kernelStart;                           
uint64_t kernelEnd;                             

} ze_kernel_timestamp_data_t;

typedef struct _ze_kernel_timestamp_result_t
{
ze_kernel_timestamp_data_t global;              
ze_kernel_timestamp_data_t context;             

} ze_kernel_timestamp_result_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryKernelTimestamp(
ze_event_handle_t hEvent,                       
ze_kernel_timestamp_result_t* dstptr            
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendQueryKernelTimestamps(
ze_command_list_handle_t hCommandList,          
uint32_t numEvents,                             
ze_event_handle_t* phEvents,                    
void* dstptr,                                   
const size_t* pOffsets,                         
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region fence
#endif
typedef uint32_t ze_fence_flags_t;
typedef enum _ze_fence_flag_t
{
ZE_FENCE_FLAG_SIGNALED = ZE_BIT(0),             
ZE_FENCE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_fence_flag_t;

typedef struct _ze_fence_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_fence_flags_t flags;                         

} ze_fence_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceCreate(
ze_command_queue_handle_t hCommandQueue,        
const ze_fence_desc_t* desc,                    
ze_fence_handle_t* phFence                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceDestroy(
ze_fence_handle_t hFence                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceHostSynchronize(
ze_fence_handle_t hFence,                       
uint64_t timeout                                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceQueryStatus(
ze_fence_handle_t hFence                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeFenceReset(
ze_fence_handle_t hFence                        
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region image
#endif
typedef uint32_t ze_image_flags_t;
typedef enum _ze_image_flag_t
{
ZE_IMAGE_FLAG_KERNEL_WRITE = ZE_BIT(0),         
ZE_IMAGE_FLAG_BIAS_UNCACHED = ZE_BIT(1),        
ZE_IMAGE_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_image_flag_t;

typedef enum _ze_image_type_t
{
ZE_IMAGE_TYPE_1D = 0,                           
ZE_IMAGE_TYPE_1DARRAY = 1,                      
ZE_IMAGE_TYPE_2D = 2,                           
ZE_IMAGE_TYPE_2DARRAY = 3,                      
ZE_IMAGE_TYPE_3D = 4,                           
ZE_IMAGE_TYPE_BUFFER = 5,                       
ZE_IMAGE_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_image_type_t;

typedef enum _ze_image_format_layout_t
{
ZE_IMAGE_FORMAT_LAYOUT_8 = 0,                   
ZE_IMAGE_FORMAT_LAYOUT_16 = 1,                  
ZE_IMAGE_FORMAT_LAYOUT_32 = 2,                  
ZE_IMAGE_FORMAT_LAYOUT_8_8 = 3,                 
ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 = 4,             
ZE_IMAGE_FORMAT_LAYOUT_16_16 = 5,               
ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 = 6,         
ZE_IMAGE_FORMAT_LAYOUT_32_32 = 7,               
ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 = 8,         
ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2 = 9,          
ZE_IMAGE_FORMAT_LAYOUT_11_11_10 = 10,           
ZE_IMAGE_FORMAT_LAYOUT_5_6_5 = 11,              
ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1 = 12,            
ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4 = 13,            
ZE_IMAGE_FORMAT_LAYOUT_Y8 = 14,                 
ZE_IMAGE_FORMAT_LAYOUT_NV12 = 15,               
ZE_IMAGE_FORMAT_LAYOUT_YUYV = 16,               
ZE_IMAGE_FORMAT_LAYOUT_VYUY = 17,               
ZE_IMAGE_FORMAT_LAYOUT_YVYU = 18,               
ZE_IMAGE_FORMAT_LAYOUT_UYVY = 19,               
ZE_IMAGE_FORMAT_LAYOUT_AYUV = 20,               
ZE_IMAGE_FORMAT_LAYOUT_P010 = 21,               
ZE_IMAGE_FORMAT_LAYOUT_Y410 = 22,               
ZE_IMAGE_FORMAT_LAYOUT_P012 = 23,               
ZE_IMAGE_FORMAT_LAYOUT_Y16 = 24,                
ZE_IMAGE_FORMAT_LAYOUT_P016 = 25,               
ZE_IMAGE_FORMAT_LAYOUT_Y216 = 26,               
ZE_IMAGE_FORMAT_LAYOUT_P216 = 27,               
ZE_IMAGE_FORMAT_LAYOUT_P8 = 28,                 
ZE_IMAGE_FORMAT_LAYOUT_YUY2 = 29,               
ZE_IMAGE_FORMAT_LAYOUT_A8P8 = 30,               
ZE_IMAGE_FORMAT_LAYOUT_IA44 = 31,               
ZE_IMAGE_FORMAT_LAYOUT_AI44 = 32,               
ZE_IMAGE_FORMAT_LAYOUT_Y416 = 33,               
ZE_IMAGE_FORMAT_LAYOUT_Y210 = 34,               
ZE_IMAGE_FORMAT_LAYOUT_I420 = 35,               
ZE_IMAGE_FORMAT_LAYOUT_YV12 = 36,               
ZE_IMAGE_FORMAT_LAYOUT_400P = 37,               
ZE_IMAGE_FORMAT_LAYOUT_422H = 38,               
ZE_IMAGE_FORMAT_LAYOUT_422V = 39,               
ZE_IMAGE_FORMAT_LAYOUT_444P = 40,               
ZE_IMAGE_FORMAT_LAYOUT_RGBP = 41,               
ZE_IMAGE_FORMAT_LAYOUT_BRGP = 42,               
ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32 = 0x7fffffff

} ze_image_format_layout_t;

typedef enum _ze_image_format_type_t
{
ZE_IMAGE_FORMAT_TYPE_UINT = 0,                  
ZE_IMAGE_FORMAT_TYPE_SINT = 1,                  
ZE_IMAGE_FORMAT_TYPE_UNORM = 2,                 
ZE_IMAGE_FORMAT_TYPE_SNORM = 3,                 
ZE_IMAGE_FORMAT_TYPE_FLOAT = 4,                 
ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_image_format_type_t;

typedef enum _ze_image_format_swizzle_t
{
ZE_IMAGE_FORMAT_SWIZZLE_R = 0,                  
ZE_IMAGE_FORMAT_SWIZZLE_G = 1,                  
ZE_IMAGE_FORMAT_SWIZZLE_B = 2,                  
ZE_IMAGE_FORMAT_SWIZZLE_A = 3,                  
ZE_IMAGE_FORMAT_SWIZZLE_0 = 4,                  
ZE_IMAGE_FORMAT_SWIZZLE_1 = 5,                  
ZE_IMAGE_FORMAT_SWIZZLE_X = 6,                  
ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32 = 0x7fffffff

} ze_image_format_swizzle_t;

typedef struct _ze_image_format_t
{
ze_image_format_layout_t layout;                
ze_image_format_type_t type;                    
ze_image_format_swizzle_t x;                    
ze_image_format_swizzle_t y;                    
ze_image_format_swizzle_t z;                    
ze_image_format_swizzle_t w;                    

} ze_image_format_t;

typedef struct _ze_image_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_image_flags_t flags;                         
ze_image_type_t type;                           
ze_image_format_t format;                       
uint64_t width;                                 
uint32_t height;                                
uint32_t depth;                                 
uint32_t arraylevels;                           
uint32_t miplevels;                             

} ze_image_desc_t;

typedef uint32_t ze_image_sampler_filter_flags_t;
typedef enum _ze_image_sampler_filter_flag_t
{
ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT = ZE_BIT(0), 
ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR = ZE_BIT(1),
ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_image_sampler_filter_flag_t;

typedef struct _ze_image_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_image_sampler_filter_flags_t samplerFilterFlags; 

} ze_image_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetProperties(
ze_device_handle_t hDevice,                     
const ze_image_desc_t* desc,                    
ze_image_properties_t* pImageProperties         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_image_desc_t* desc,                    
ze_image_handle_t* phImage                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageDestroy(
ze_image_handle_t hImage                        
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region memory
#endif
typedef uint32_t ze_device_mem_alloc_flags_t;
typedef enum _ze_device_mem_alloc_flag_t
{
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = ZE_BIT(0),   
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = ZE_BIT(1), 
ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = ZE_BIT(2),
ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_mem_alloc_flag_t;

typedef struct _ze_device_mem_alloc_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_device_mem_alloc_flags_t flags;              
uint32_t ordinal;                               

} ze_device_mem_alloc_desc_t;

typedef uint32_t ze_host_mem_alloc_flags_t;
typedef enum _ze_host_mem_alloc_flag_t
{
ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = ZE_BIT(0), 
ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = ZE_BIT(1),   
ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = ZE_BIT(2), 
ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = ZE_BIT(3),  
ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_host_mem_alloc_flag_t;

typedef struct _ze_host_mem_alloc_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_host_mem_alloc_flags_t flags;                

} ze_host_mem_alloc_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocShared(
ze_context_handle_t hContext,                   
const ze_device_mem_alloc_desc_t* device_desc,  
const ze_host_mem_alloc_desc_t* host_desc,      
size_t size,                                    
size_t alignment,                               
ze_device_handle_t hDevice,                     
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocDevice(
ze_context_handle_t hContext,                   
const ze_device_mem_alloc_desc_t* device_desc,  
size_t size,                                    
size_t alignment,                               
ze_device_handle_t hDevice,                     
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemAllocHost(
ze_context_handle_t hContext,                   
const ze_host_mem_alloc_desc_t* host_desc,      
size_t size,                                    
size_t alignment,                               
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemFree(
ze_context_handle_t hContext,                   
void* ptr                                       
);

typedef enum _ze_memory_type_t
{
ZE_MEMORY_TYPE_UNKNOWN = 0,                     
ZE_MEMORY_TYPE_HOST = 1,                        
ZE_MEMORY_TYPE_DEVICE = 2,                      
ZE_MEMORY_TYPE_SHARED = 3,                      
ZE_MEMORY_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_memory_type_t;

typedef struct _ze_memory_allocation_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_memory_type_t type;                          
uint64_t id;                                    
uint64_t pageSize;                              

} ze_memory_allocation_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAllocProperties(
ze_context_handle_t hContext,                   
const void* ptr,                                
ze_memory_allocation_properties_t* pMemAllocProperties, 
ze_device_handle_t* phDevice                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetAddressRange(
ze_context_handle_t hContext,                   
const void* ptr,                                
void** pBase,                                   
size_t* pSize                                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemGetIpcHandle(
ze_context_handle_t hContext,                   
const void* ptr,                                
ze_ipc_mem_handle_t* pIpcHandle                 
);

typedef uint32_t ze_ipc_memory_flags_t;
typedef enum _ze_ipc_memory_flag_t
{
ZE_IPC_MEMORY_FLAG_BIAS_CACHED = ZE_BIT(0),     
ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED = ZE_BIT(1),   
ZE_IPC_MEMORY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_ipc_memory_flag_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemOpenIpcHandle(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
ze_ipc_mem_handle_t handle,                     
ze_ipc_memory_flags_t flags,                    
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemCloseIpcHandle(
ze_context_handle_t hContext,                   
const void* ptr                                 
);

typedef struct _ze_external_memory_export_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_external_memory_type_flags_t flags;          

} ze_external_memory_export_desc_t;

typedef struct _ze_external_memory_import_fd_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_external_memory_type_flags_t flags;          
int fd;                                         

} ze_external_memory_import_fd_t;

typedef struct _ze_external_memory_export_fd_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_external_memory_type_flags_t flags;          
int fd;                                         

} ze_external_memory_export_fd_t;

typedef struct _ze_external_memory_import_win32_handle_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_external_memory_type_flags_t flags;          
void* handle;                                   
const void* name;                               

} ze_external_memory_import_win32_handle_t;

typedef struct _ze_external_memory_export_win32_handle_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_external_memory_type_flags_t flags;          
void* handle;                                   

} ze_external_memory_export_win32_handle_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region module
#endif
typedef enum _ze_module_format_t
{
ZE_MODULE_FORMAT_IL_SPIRV = 0,                  
ZE_MODULE_FORMAT_NATIVE = 1,                    
ZE_MODULE_FORMAT_FORCE_UINT32 = 0x7fffffff

} ze_module_format_t;

typedef struct _ze_module_constants_t
{
uint32_t numConstants;                          
const uint32_t* pConstantIds;                   
const void** pConstantValues;                   

} ze_module_constants_t;

typedef struct _ze_module_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_module_format_t format;                      
size_t inputSize;                               
const uint8_t* pInputModule;                    
const char* pBuildFlags;                        
const ze_module_constants_t* pConstants;        

} ze_module_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_module_desc_t* desc,                   
ze_module_handle_t* phModule,                   
ze_module_build_log_handle_t* phBuildLog        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleDestroy(
ze_module_handle_t hModule                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleDynamicLink(
uint32_t numModules,                            
ze_module_handle_t* phModules,                  
ze_module_build_log_handle_t* phLinkLog         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogDestroy(
ze_module_build_log_handle_t hModuleBuildLog    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleBuildLogGetString(
ze_module_build_log_handle_t hModuleBuildLog,   
size_t* pSize,                                  
char* pBuildLog                                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetNativeBinary(
ze_module_handle_t hModule,                     
size_t* pSize,                                  
uint8_t* pModuleNativeBinary                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetGlobalPointer(
ze_module_handle_t hModule,                     
const char* pGlobalName,                        
size_t* pSize,                                  
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetKernelNames(
ze_module_handle_t hModule,                     
uint32_t* pCount,                               
const char** pNames                             
);

typedef uint32_t ze_module_property_flags_t;
typedef enum _ze_module_property_flag_t
{
ZE_MODULE_PROPERTY_FLAG_IMPORTS = ZE_BIT(0),    
ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_module_property_flag_t;

typedef struct _ze_module_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_module_property_flags_t flags;               

} ze_module_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetProperties(
ze_module_handle_t hModule,                     
ze_module_properties_t* pModuleProperties       
);

typedef uint32_t ze_kernel_flags_t;
typedef enum _ze_kernel_flag_t
{
ZE_KERNEL_FLAG_FORCE_RESIDENCY = ZE_BIT(0),     
ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY = ZE_BIT(1),  
ZE_KERNEL_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_kernel_flag_t;

typedef struct _ze_kernel_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_kernel_flags_t flags;                        
const char* pKernelName;                        

} ze_kernel_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelCreate(
ze_module_handle_t hModule,                     
const ze_kernel_desc_t* desc,                   
ze_kernel_handle_t* phKernel                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelDestroy(
ze_kernel_handle_t hKernel                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleGetFunctionPointer(
ze_module_handle_t hModule,                     
const char* pFunctionName,                      
void** pfnFunction                              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetGroupSize(
ze_kernel_handle_t hKernel,                     
uint32_t groupSizeX,                            
uint32_t groupSizeY,                            
uint32_t groupSizeZ                             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSuggestGroupSize(
ze_kernel_handle_t hKernel,                     
uint32_t globalSizeX,                           
uint32_t globalSizeY,                           
uint32_t globalSizeZ,                           
uint32_t* groupSizeX,                           
uint32_t* groupSizeY,                           
uint32_t* groupSizeZ                            
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSuggestMaxCooperativeGroupCount(
ze_kernel_handle_t hKernel,                     
uint32_t* totalGroupCount                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetArgumentValue(
ze_kernel_handle_t hKernel,                     
uint32_t argIndex,                              
size_t argSize,                                 
const void* pArgValue                           
);

typedef uint32_t ze_kernel_indirect_access_flags_t;
typedef enum _ze_kernel_indirect_access_flag_t
{
ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = ZE_BIT(0),
ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = ZE_BIT(1),  
ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = ZE_BIT(2),  
ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_kernel_indirect_access_flag_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetIndirectAccess(
ze_kernel_handle_t hKernel,                     
ze_kernel_indirect_access_flags_t flags         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetIndirectAccess(
ze_kernel_handle_t hKernel,                     
ze_kernel_indirect_access_flags_t* pFlags       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetSourceAttributes(
ze_kernel_handle_t hKernel,                     
uint32_t* pSize,                                
char** pString                                  
);

typedef uint32_t ze_cache_config_flags_t;
typedef enum _ze_cache_config_flag_t
{
ZE_CACHE_CONFIG_FLAG_LARGE_SLM = ZE_BIT(0),     
ZE_CACHE_CONFIG_FLAG_LARGE_DATA = ZE_BIT(1),    
ZE_CACHE_CONFIG_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_cache_config_flag_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetCacheConfig(
ze_kernel_handle_t hKernel,                     
ze_cache_config_flags_t flags                   
);

#ifndef ZE_MAX_KERNEL_UUID_SIZE
#define ZE_MAX_KERNEL_UUID_SIZE  16
#endif 

#ifndef ZE_MAX_MODULE_UUID_SIZE
#define ZE_MAX_MODULE_UUID_SIZE  16
#endif 

typedef struct _ze_kernel_uuid_t
{
uint8_t kid[ZE_MAX_KERNEL_UUID_SIZE];           
uint8_t mid[ZE_MAX_MODULE_UUID_SIZE];           

} ze_kernel_uuid_t;

typedef struct _ze_kernel_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint32_t numKernelArgs;                         
uint32_t requiredGroupSizeX;                    
uint32_t requiredGroupSizeY;                    
uint32_t requiredGroupSizeZ;                    
uint32_t requiredNumSubGroups;                  
uint32_t requiredSubgroupSize;                  
uint32_t maxSubgroupSize;                       
uint32_t maxNumSubgroups;                       
uint32_t localMemSize;                          
uint32_t privateMemSize;                        
uint32_t spillMemSize;                          
ze_kernel_uuid_t uuid;                          

} ze_kernel_properties_t;

typedef struct _ze_kernel_preferred_group_size_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint32_t preferredMultiple;                     

} ze_kernel_preferred_group_size_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetProperties(
ze_kernel_handle_t hKernel,                     
ze_kernel_properties_t* pKernelProperties       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelGetName(
ze_kernel_handle_t hKernel,                     
size_t* pSize,                                  
char* pName                                     
);

typedef struct _ze_group_count_t
{
uint32_t groupCountX;                           
uint32_t groupCountY;                           
uint32_t groupCountZ;                           

} ze_group_count_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernel(
ze_command_list_handle_t hCommandList,          
ze_kernel_handle_t hKernel,                     
const ze_group_count_t* pLaunchFuncArgs,        
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchCooperativeKernel(
ze_command_list_handle_t hCommandList,          
ze_kernel_handle_t hKernel,                     
const ze_group_count_t* pLaunchFuncArgs,        
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernelIndirect(
ze_command_list_handle_t hCommandList,          
ze_kernel_handle_t hKernel,                     
const ze_group_count_t* pLaunchArgumentsBuffer, 
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchMultipleKernelsIndirect(
ze_command_list_handle_t hCommandList,          
uint32_t numKernels,                            
ze_kernel_handle_t* phKernels,                  
const uint32_t* pCountBuffer,                   
const ze_group_count_t* pLaunchArgumentsBuffer, 
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region program
#endif
#ifndef ZE_MODULE_PROGRAM_EXP_NAME
#define ZE_MODULE_PROGRAM_EXP_NAME  "ZE_experimental_module_program"
#endif 

typedef enum _ze_module_program_exp_version_t
{
ZE_MODULE_PROGRAM_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_module_program_exp_version_t;

typedef struct _ze_module_program_exp_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t count;                                 
const size_t* inputSizes;                       
const uint8_t** pInputModules;                  
const char** pBuildFlags;                       
const ze_module_constants_t** pConstants;       

} ze_module_program_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region raytracing
#endif
#ifndef ZE_RAYTRACING_EXT_NAME
#define ZE_RAYTRACING_EXT_NAME  "ZE_extension_raytracing"
#endif 

typedef enum _ze_raytracing_ext_version_t
{
ZE_RAYTRACING_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_RAYTRACING_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_raytracing_ext_version_t;

typedef uint32_t ze_device_raytracing_ext_flags_t;
typedef enum _ze_device_raytracing_ext_flag_t
{
ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY = ZE_BIT(0), 
ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_raytracing_ext_flag_t;

typedef struct _ze_device_raytracing_ext_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_raytracing_ext_flags_t flags;         
uint32_t maxBVHLevels;                          

} ze_device_raytracing_ext_properties_t;

typedef uint32_t ze_raytracing_mem_alloc_ext_flags_t;
typedef enum _ze_raytracing_mem_alloc_ext_flag_t
{
ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD = ZE_BIT(0),   
ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_raytracing_mem_alloc_ext_flag_t;

typedef struct _ze_raytracing_mem_alloc_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_raytracing_mem_alloc_ext_flags_t flags;      

} ze_raytracing_mem_alloc_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region residency
#endif
ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextMakeMemoryResident(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
void* ptr,                                      
size_t size                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextEvictMemory(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
void* ptr,                                      
size_t size                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextMakeImageResident(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
ze_image_handle_t hImage                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeContextEvictImage(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
ze_image_handle_t hImage                        
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region sampler
#endif
typedef enum _ze_sampler_address_mode_t
{
ZE_SAMPLER_ADDRESS_MODE_NONE = 0,               
ZE_SAMPLER_ADDRESS_MODE_REPEAT = 1,             
ZE_SAMPLER_ADDRESS_MODE_CLAMP = 2,              
ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3,    
ZE_SAMPLER_ADDRESS_MODE_MIRROR = 4,             
ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32 = 0x7fffffff

} ze_sampler_address_mode_t;

typedef enum _ze_sampler_filter_mode_t
{
ZE_SAMPLER_FILTER_MODE_NEAREST = 0,             
ZE_SAMPLER_FILTER_MODE_LINEAR = 1,              
ZE_SAMPLER_FILTER_MODE_FORCE_UINT32 = 0x7fffffff

} ze_sampler_filter_mode_t;

typedef struct _ze_sampler_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_sampler_address_mode_t addressMode;          
ze_sampler_filter_mode_t filterMode;            
ze_bool_t isNormalized;                         

} ze_sampler_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeSamplerCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_sampler_desc_t* desc,                  
ze_sampler_handle_t* phSampler                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeSamplerDestroy(
ze_sampler_handle_t hSampler                    
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region virtual
#endif
typedef enum _ze_memory_access_attribute_t
{
ZE_MEMORY_ACCESS_ATTRIBUTE_NONE = 0,            
ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1,       
ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY = 2,        
ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32 = 0x7fffffff

} ze_memory_access_attribute_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemReserve(
ze_context_handle_t hContext,                   
const void* pStart,                             
size_t size,                                    
void** pptr                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemFree(
ze_context_handle_t hContext,                   
const void* ptr,                                
size_t size                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemQueryPageSize(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
size_t size,                                    
size_t* pagesize                                
);

typedef uint32_t ze_physical_mem_flags_t;
typedef enum _ze_physical_mem_flag_t
{
ZE_PHYSICAL_MEM_FLAG_TBD = ZE_BIT(0),           
ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_physical_mem_flag_t;

typedef struct _ze_physical_mem_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_physical_mem_flags_t flags;                  
size_t size;                                    

} ze_physical_mem_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zePhysicalMemCreate(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
ze_physical_mem_desc_t* desc,                   
ze_physical_mem_handle_t* phPhysicalMemory      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zePhysicalMemDestroy(
ze_context_handle_t hContext,                   
ze_physical_mem_handle_t hPhysicalMemory        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemMap(
ze_context_handle_t hContext,                   
const void* ptr,                                
size_t size,                                    
ze_physical_mem_handle_t hPhysicalMemory,       
size_t offset,                                  
ze_memory_access_attribute_t access             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemUnmap(
ze_context_handle_t hContext,                   
const void* ptr,                                
size_t size                                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemSetAccessAttribute(
ze_context_handle_t hContext,                   
const void* ptr,                                
size_t size,                                    
ze_memory_access_attribute_t access             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeVirtualMemGetAccessAttribute(
ze_context_handle_t hContext,                   
const void* ptr,                                
size_t size,                                    
ze_memory_access_attribute_t* access,           
size_t* outSize                                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region floatAtomics
#endif
#ifndef ZE_FLOAT_ATOMICS_EXT_NAME
#define ZE_FLOAT_ATOMICS_EXT_NAME  "ZE_extension_float_atomics"
#endif 

typedef enum _ze_float_atomics_ext_version_t
{
ZE_FLOAT_ATOMICS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_float_atomics_ext_version_t;

typedef uint32_t ze_device_fp_atomic_ext_flags_t;
typedef enum _ze_device_fp_atomic_ext_flag_t
{
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE = ZE_BIT(0), 
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD = ZE_BIT(1),
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX = ZE_BIT(2),
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE = ZE_BIT(16), 
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD = ZE_BIT(17),
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX = ZE_BIT(18),
ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_device_fp_atomic_ext_flag_t;

typedef struct _ze_float_atomic_ext_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_device_fp_atomic_ext_flags_t fp16Flags;      
ze_device_fp_atomic_ext_flags_t fp32Flags;      
ze_device_fp_atomic_ext_flags_t fp64Flags;      

} ze_float_atomic_ext_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region globaloffset
#endif
#ifndef ZE_GLOBAL_OFFSET_EXP_NAME
#define ZE_GLOBAL_OFFSET_EXP_NAME  "ZE_experimental_global_offset"
#endif 

typedef enum _ze_global_offset_exp_version_t
{
ZE_GLOBAL_OFFSET_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_global_offset_exp_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSetGlobalOffsetExp(
ze_kernel_handle_t hKernel,                     
uint32_t offsetX,                               
uint32_t offsetY,                               
uint32_t offsetZ                                
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region relaxedAllocLimits
#endif
#ifndef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
#define ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME  "ZE_experimental_relaxed_allocation_limits"
#endif 

typedef enum _ze_relaxed_allocation_limits_exp_version_t
{
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_relaxed_allocation_limits_exp_version_t;

typedef uint32_t ze_relaxed_allocation_limits_exp_flags_t;
typedef enum _ze_relaxed_allocation_limits_exp_flag_t
{
ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = ZE_BIT(0), 
ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_relaxed_allocation_limits_exp_flag_t;

typedef struct _ze_relaxed_allocation_limits_exp_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_relaxed_allocation_limits_exp_flags_t flags; 

} ze_relaxed_allocation_limits_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region cacheReservation
#endif
#ifndef ZE_CACHE_RESERVATION_EXT_NAME
#define ZE_CACHE_RESERVATION_EXT_NAME  "ZE_extension_cache_reservation"
#endif 

typedef enum _ze_cache_reservation_ext_version_t
{
ZE_CACHE_RESERVATION_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_cache_reservation_ext_version_t;

typedef enum _ze_cache_ext_region_t
{
ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT = 0,
ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION = 1,
ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION = 2,   
ZE_CACHE_EXT_REGION_FORCE_UINT32 = 0x7fffffff

} ze_cache_ext_region_t;

typedef struct _ze_cache_reservation_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
size_t maxCacheReservationSize;                 

} ze_cache_reservation_ext_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceReserveCacheExt(
ze_device_handle_t hDevice,                     
size_t cacheLevel,                              
size_t cacheReservationSize                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDeviceSetCacheAdviceExt(
ze_device_handle_t hDevice,                     
void* ptr,                                      
size_t regionSize,                              
ze_cache_ext_region_t cacheRegion               
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region eventquerytimestamps
#endif
#ifndef ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME
#define ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME  "ZE_experimental_event_query_timestamps"
#endif 

typedef enum _ze_event_query_timestamps_exp_version_t
{
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_event_query_timestamps_exp_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeEventQueryTimestampsExp(
ze_event_handle_t hEvent,                       
ze_device_handle_t hDevice,                     
uint32_t* pCount,                               
ze_kernel_timestamp_result_t* pTimestamps       
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region imagememoryproperties
#endif
#ifndef ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME
#define ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME  "ZE_experimental_image_memory_properties"
#endif 

typedef enum _ze_image_memory_properties_exp_version_t
{
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),   
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),   
ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_memory_properties_exp_version_t;

typedef struct _ze_image_memory_properties_exp_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint64_t size;                                  
uint64_t rowPitch;                              
uint64_t slicePitch;                            

} ze_image_memory_properties_exp_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetMemoryPropertiesExp(
ze_image_handle_t hImage,                       
ze_image_memory_properties_exp_t* pMemoryProperties 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region imageview
#endif
#ifndef ZE_IMAGE_VIEW_EXP_NAME
#define ZE_IMAGE_VIEW_EXP_NAME  "ZE_experimental_image_view"
#endif 

typedef enum _ze_image_view_exp_version_t
{
ZE_IMAGE_VIEW_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_IMAGE_VIEW_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_exp_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageViewCreateExp(
ze_context_handle_t hContext,                   
ze_device_handle_t hDevice,                     
const ze_image_desc_t* desc,                    
ze_image_handle_t hImage,                       
ze_image_handle_t* phImageView                  
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region imageviewplanar
#endif
#ifndef ZE_IMAGE_VIEW_PLANAR_EXP_NAME
#define ZE_IMAGE_VIEW_PLANAR_EXP_NAME  "ZE_experimental_image_view_planar"
#endif 

typedef enum _ze_image_view_planar_exp_version_t
{
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_view_planar_exp_version_t;

typedef struct _ze_image_view_planar_exp_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t planeIndex;                            

} ze_image_view_planar_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region kernelSchedulingHints
#endif
#ifndef ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME
#define ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME  "ZE_experimental_scheduling_hints"
#endif 

typedef enum _ze_scheduling_hints_exp_version_t
{
ZE_SCHEDULING_HINTS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_scheduling_hints_exp_version_t;

typedef uint32_t ze_scheduling_hint_exp_flags_t;
typedef enum _ze_scheduling_hint_exp_flag_t
{
ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST = ZE_BIT(0),   
ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN = ZE_BIT(1),
ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN = ZE_BIT(2),
ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_scheduling_hint_exp_flag_t;

typedef struct _ze_scheduling_hint_exp_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_scheduling_hint_exp_flags_t schedulingHintFlags; 

} ze_scheduling_hint_exp_properties_t;

typedef struct _ze_scheduling_hint_exp_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_scheduling_hint_exp_flags_t flags;           

} ze_scheduling_hint_exp_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeKernelSchedulingHintExp(
ze_kernel_handle_t hKernel,                     
ze_scheduling_hint_exp_desc_t* pHint            
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region linkonceodr
#endif
#ifndef ZE_LINKONCE_ODR_EXT_NAME
#define ZE_LINKONCE_ODR_EXT_NAME  "ZE_extension_linkonce_odr"
#endif 

typedef enum _ze_linkonce_odr_ext_version_t
{
ZE_LINKONCE_ODR_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_LINKONCE_ODR_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_linkonce_odr_ext_version_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region powersavinghint
#endif
#ifndef ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME
#define ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME  "ZE_experimental_power_saving_hint"
#endif 

typedef enum _ze_power_saving_hint_exp_version_t
{
ZE_POWER_SAVING_HINT_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), 
ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), 
ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_power_saving_hint_exp_version_t;

typedef enum _ze_power_saving_hint_type_t
{
ZE_POWER_SAVING_HINT_TYPE_MIN = 0,              
ZE_POWER_SAVING_HINT_TYPE_MAX = 100,            
ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32 = 0x7fffffff

} ze_power_saving_hint_type_t;

typedef struct _ze_context_power_saving_hint_exp_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t hint;                                  

} ze_context_power_saving_hint_exp_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region subgroups
#endif
#ifndef ZE_SUBGROUPS_EXT_NAME
#define ZE_SUBGROUPS_EXT_NAME  "ZE_extension_subgroups"
#endif 

typedef enum _ze_subgroup_ext_version_t
{
ZE_SUBGROUP_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SUBGROUP_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_subgroup_ext_version_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region EUCount
#endif
#ifndef ZE_EU_COUNT_EXT_NAME
#define ZE_EU_COUNT_EXT_NAME  "ZE_extension_eu_count"
#endif 

typedef enum _ze_eu_count_ext_version_t
{
ZE_EU_COUNT_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_EU_COUNT_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_eu_count_ext_version_t;

typedef struct _ze_eu_count_ext_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
uint32_t numTotalEUs;                           

} ze_eu_count_ext_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region PCIProperties
#endif
#ifndef ZE_PCI_PROPERTIES_EXT_NAME
#define ZE_PCI_PROPERTIES_EXT_NAME  "ZE_extension_pci_properties"
#endif 

typedef enum _ze_pci_properties_ext_version_t
{
ZE_PCI_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_pci_properties_ext_version_t;

typedef struct _ze_pci_address_ext_t
{
uint32_t domain;                                
uint32_t bus;                                   
uint32_t device;                                
uint32_t function;                              

} ze_pci_address_ext_t;

typedef struct _ze_pci_speed_ext_t
{
int32_t genVersion;                             
int32_t width;                                  
int64_t maxBandwidth;                           

} ze_pci_speed_ext_t;

typedef struct _ze_pci_ext_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_pci_address_ext_t address;                   
ze_pci_speed_ext_t maxSpeed;                    

} ze_pci_ext_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeDevicePciGetPropertiesExt(
ze_device_handle_t hDevice,                     
ze_pci_ext_properties_t* pPciProperties         
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region SRGB
#endif
#ifndef ZE_SRGB_EXT_NAME
#define ZE_SRGB_EXT_NAME  "ZE_extension_srgb"
#endif 

typedef enum _ze_srgb_ext_version_t
{
ZE_SRGB_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SRGB_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_SRGB_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_srgb_ext_version_t;

typedef struct _ze_srgb_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_bool_t sRGB;                                 

} ze_srgb_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region imageCopy
#endif
#ifndef ZE_IMAGE_COPY_EXT_NAME
#define ZE_IMAGE_COPY_EXT_NAME  "ZE_extension_image_copy"
#endif 

typedef enum _ze_image_copy_ext_version_t
{
ZE_IMAGE_COPY_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_IMAGE_COPY_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_copy_ext_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyToMemoryExt(
ze_command_list_handle_t hCommandList,          
void* dstptr,                                   
ze_image_handle_t hSrcImage,                    
const ze_image_region_t* pSrcRegion,            
uint32_t destRowPitch,                          
uint32_t destSlicePitch,                        
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendImageCopyFromMemoryExt(
ze_command_list_handle_t hCommandList,          
ze_image_handle_t hDstImage,                    
const void* srcptr,                             
const ze_image_region_t* pDstRegion,            
uint32_t srcRowPitch,                           
uint32_t srcSlicePitch,                         
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region imageQueryAllocProperties
#endif
#ifndef ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME
#define ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME  "ZE_extension_image_query_alloc_properties"
#endif 

typedef enum _ze_image_query_alloc_properties_ext_version_t
{
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_image_query_alloc_properties_ext_version_t;

typedef struct _ze_image_allocation_ext_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
uint64_t id;                                    

} ze_image_allocation_ext_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeImageGetAllocPropertiesExt(
ze_context_handle_t hContext,                   
ze_image_handle_t hImage,                       
ze_image_allocation_ext_properties_t* pImageAllocProperties 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region linkageInspection
#endif
#ifndef ZE_LINKAGE_INSPECTION_EXT_NAME
#define ZE_LINKAGE_INSPECTION_EXT_NAME  "ZE_extension_linkage_inspection"
#endif 

typedef enum _ze_linkage_inspection_ext_version_t
{
ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_linkage_inspection_ext_version_t;

typedef uint32_t ze_linkage_inspection_ext_flags_t;
typedef enum _ze_linkage_inspection_ext_flag_t
{
ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS = ZE_BIT(0), 
ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS = ZE_BIT(1),
ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS = ZE_BIT(2), 
ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_linkage_inspection_ext_flag_t;

typedef struct _ze_linkage_inspection_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_linkage_inspection_ext_flags_t flags;        

} ze_linkage_inspection_ext_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeModuleInspectLinkageExt(
ze_linkage_inspection_ext_desc_t* pInspectDesc, 
uint32_t numModules,                            
ze_module_handle_t* phModules,                  
ze_module_build_log_handle_t* phLog             
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region memoryCompressionHints
#endif
#ifndef ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME
#define ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME  "ZE_extension_memory_compression_hints"
#endif 

typedef enum _ze_memory_compression_hints_ext_version_t
{
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_memory_compression_hints_ext_version_t;

typedef uint32_t ze_memory_compression_hints_ext_flags_t;
typedef enum _ze_memory_compression_hints_ext_flag_t
{
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED = ZE_BIT(0),
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED = ZE_BIT(1),  
ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_memory_compression_hints_ext_flag_t;

typedef struct _ze_memory_compression_hints_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_memory_compression_hints_ext_flags_t flags;  

} ze_memory_compression_hints_ext_desc_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region memoryFreePolicies
#endif
#ifndef ZE_MEMORY_FREE_POLICIES_EXT_NAME
#define ZE_MEMORY_FREE_POLICIES_EXT_NAME  "ZE_extension_memory_free_policies"
#endif 

typedef enum _ze_memory_free_policies_ext_version_t
{
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_memory_free_policies_ext_version_t;

typedef uint32_t ze_driver_memory_free_policy_ext_flags_t;
typedef enum _ze_driver_memory_free_policy_ext_flag_t
{
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE = ZE_BIT(0),
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE = ZE_BIT(1),   
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_driver_memory_free_policy_ext_flag_t;

typedef struct _ze_driver_memory_free_ext_properties_t
{
ze_structure_type_t stype;                      
void* pNext;                                    
ze_driver_memory_free_policy_ext_flags_t freePolicies;  

} ze_driver_memory_free_ext_properties_t;

typedef struct _ze_memory_free_ext_desc_t
{
ze_structure_type_t stype;                      
const void* pNext;                              
ze_driver_memory_free_policy_ext_flags_t freePolicy;

} ze_memory_free_ext_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeMemFreeExt(
ze_context_handle_t hContext,                   
const ze_memory_free_ext_desc_t* pMemFreeDesc,  
void* ptr                                       
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region callbacks
#endif
typedef struct _ze_init_params_t
{
ze_init_flags_t* pflags;
} ze_init_params_t;

typedef void (ZE_APICALL *ze_pfnInitCb_t)(
ze_init_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_global_callbacks_t
{
ze_pfnInitCb_t                                                  pfnInitCb;
} ze_global_callbacks_t;

typedef struct _ze_driver_get_params_t
{
uint32_t** ppCount;
ze_driver_handle_t** pphDrivers;
} ze_driver_get_params_t;

typedef void (ZE_APICALL *ze_pfnDriverGetCb_t)(
ze_driver_get_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_driver_get_api_version_params_t
{
ze_driver_handle_t* phDriver;
ze_api_version_t** pversion;
} ze_driver_get_api_version_params_t;

typedef void (ZE_APICALL *ze_pfnDriverGetApiVersionCb_t)(
ze_driver_get_api_version_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_driver_get_properties_params_t
{
ze_driver_handle_t* phDriver;
ze_driver_properties_t** ppDriverProperties;
} ze_driver_get_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDriverGetPropertiesCb_t)(
ze_driver_get_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_driver_get_ipc_properties_params_t
{
ze_driver_handle_t* phDriver;
ze_driver_ipc_properties_t** ppIpcProperties;
} ze_driver_get_ipc_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDriverGetIpcPropertiesCb_t)(
ze_driver_get_ipc_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_driver_get_extension_properties_params_t
{
ze_driver_handle_t* phDriver;
uint32_t** ppCount;
ze_driver_extension_properties_t** ppExtensionProperties;
} ze_driver_get_extension_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDriverGetExtensionPropertiesCb_t)(
ze_driver_get_extension_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_driver_callbacks_t
{
ze_pfnDriverGetCb_t                                             pfnGetCb;
ze_pfnDriverGetApiVersionCb_t                                   pfnGetApiVersionCb;
ze_pfnDriverGetPropertiesCb_t                                   pfnGetPropertiesCb;
ze_pfnDriverGetIpcPropertiesCb_t                                pfnGetIpcPropertiesCb;
ze_pfnDriverGetExtensionPropertiesCb_t                          pfnGetExtensionPropertiesCb;
} ze_driver_callbacks_t;

typedef struct _ze_device_get_params_t
{
ze_driver_handle_t* phDriver;
uint32_t** ppCount;
ze_device_handle_t** pphDevices;
} ze_device_get_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetCb_t)(
ze_device_get_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_sub_devices_params_t
{
ze_device_handle_t* phDevice;
uint32_t** ppCount;
ze_device_handle_t** pphSubdevices;
} ze_device_get_sub_devices_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetSubDevicesCb_t)(
ze_device_get_sub_devices_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_properties_t** ppDeviceProperties;
} ze_device_get_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetPropertiesCb_t)(
ze_device_get_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_compute_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_compute_properties_t** ppComputeProperties;
} ze_device_get_compute_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetComputePropertiesCb_t)(
ze_device_get_compute_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_module_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_module_properties_t** ppModuleProperties;
} ze_device_get_module_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetModulePropertiesCb_t)(
ze_device_get_module_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_command_queue_group_properties_params_t
{
ze_device_handle_t* phDevice;
uint32_t** ppCount;
ze_command_queue_group_properties_t** ppCommandQueueGroupProperties;
} ze_device_get_command_queue_group_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t)(
ze_device_get_command_queue_group_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_memory_properties_params_t
{
ze_device_handle_t* phDevice;
uint32_t** ppCount;
ze_device_memory_properties_t** ppMemProperties;
} ze_device_get_memory_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetMemoryPropertiesCb_t)(
ze_device_get_memory_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_memory_access_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_memory_access_properties_t** ppMemAccessProperties;
} ze_device_get_memory_access_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetMemoryAccessPropertiesCb_t)(
ze_device_get_memory_access_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_cache_properties_params_t
{
ze_device_handle_t* phDevice;
uint32_t** ppCount;
ze_device_cache_properties_t** ppCacheProperties;
} ze_device_get_cache_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetCachePropertiesCb_t)(
ze_device_get_cache_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_image_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_image_properties_t** ppImageProperties;
} ze_device_get_image_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetImagePropertiesCb_t)(
ze_device_get_image_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_external_memory_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_external_memory_properties_t** ppExternalMemoryProperties;
} ze_device_get_external_memory_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetExternalMemoryPropertiesCb_t)(
ze_device_get_external_memory_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_p2_p_properties_params_t
{
ze_device_handle_t* phDevice;
ze_device_handle_t* phPeerDevice;
ze_device_p2p_properties_t** ppP2PProperties;
} ze_device_get_p2_p_properties_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetP2PPropertiesCb_t)(
ze_device_get_p2_p_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_can_access_peer_params_t
{
ze_device_handle_t* phDevice;
ze_device_handle_t* phPeerDevice;
ze_bool_t** pvalue;
} ze_device_can_access_peer_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceCanAccessPeerCb_t)(
ze_device_can_access_peer_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_get_status_params_t
{
ze_device_handle_t* phDevice;
} ze_device_get_status_params_t;

typedef void (ZE_APICALL *ze_pfnDeviceGetStatusCb_t)(
ze_device_get_status_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_device_callbacks_t
{
ze_pfnDeviceGetCb_t                                             pfnGetCb;
ze_pfnDeviceGetSubDevicesCb_t                                   pfnGetSubDevicesCb;
ze_pfnDeviceGetPropertiesCb_t                                   pfnGetPropertiesCb;
ze_pfnDeviceGetComputePropertiesCb_t                            pfnGetComputePropertiesCb;
ze_pfnDeviceGetModulePropertiesCb_t                             pfnGetModulePropertiesCb;
ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t                  pfnGetCommandQueueGroupPropertiesCb;
ze_pfnDeviceGetMemoryPropertiesCb_t                             pfnGetMemoryPropertiesCb;
ze_pfnDeviceGetMemoryAccessPropertiesCb_t                       pfnGetMemoryAccessPropertiesCb;
ze_pfnDeviceGetCachePropertiesCb_t                              pfnGetCachePropertiesCb;
ze_pfnDeviceGetImagePropertiesCb_t                              pfnGetImagePropertiesCb;
ze_pfnDeviceGetExternalMemoryPropertiesCb_t                     pfnGetExternalMemoryPropertiesCb;
ze_pfnDeviceGetP2PPropertiesCb_t                                pfnGetP2PPropertiesCb;
ze_pfnDeviceCanAccessPeerCb_t                                   pfnCanAccessPeerCb;
ze_pfnDeviceGetStatusCb_t                                       pfnGetStatusCb;
} ze_device_callbacks_t;

typedef struct _ze_context_create_params_t
{
ze_driver_handle_t* phDriver;
const ze_context_desc_t** pdesc;
ze_context_handle_t** pphContext;
} ze_context_create_params_t;

typedef void (ZE_APICALL *ze_pfnContextCreateCb_t)(
ze_context_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_destroy_params_t
{
ze_context_handle_t* phContext;
} ze_context_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnContextDestroyCb_t)(
ze_context_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_get_status_params_t
{
ze_context_handle_t* phContext;
} ze_context_get_status_params_t;

typedef void (ZE_APICALL *ze_pfnContextGetStatusCb_t)(
ze_context_get_status_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_system_barrier_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
} ze_context_system_barrier_params_t;

typedef void (ZE_APICALL *ze_pfnContextSystemBarrierCb_t)(
ze_context_system_barrier_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_make_memory_resident_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
void** pptr;
size_t* psize;
} ze_context_make_memory_resident_params_t;

typedef void (ZE_APICALL *ze_pfnContextMakeMemoryResidentCb_t)(
ze_context_make_memory_resident_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_evict_memory_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
void** pptr;
size_t* psize;
} ze_context_evict_memory_params_t;

typedef void (ZE_APICALL *ze_pfnContextEvictMemoryCb_t)(
ze_context_evict_memory_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_make_image_resident_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
ze_image_handle_t* phImage;
} ze_context_make_image_resident_params_t;

typedef void (ZE_APICALL *ze_pfnContextMakeImageResidentCb_t)(
ze_context_make_image_resident_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_evict_image_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
ze_image_handle_t* phImage;
} ze_context_evict_image_params_t;

typedef void (ZE_APICALL *ze_pfnContextEvictImageCb_t)(
ze_context_evict_image_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_context_callbacks_t
{
ze_pfnContextCreateCb_t                                         pfnCreateCb;
ze_pfnContextDestroyCb_t                                        pfnDestroyCb;
ze_pfnContextGetStatusCb_t                                      pfnGetStatusCb;
ze_pfnContextSystemBarrierCb_t                                  pfnSystemBarrierCb;
ze_pfnContextMakeMemoryResidentCb_t                             pfnMakeMemoryResidentCb;
ze_pfnContextEvictMemoryCb_t                                    pfnEvictMemoryCb;
ze_pfnContextMakeImageResidentCb_t                              pfnMakeImageResidentCb;
ze_pfnContextEvictImageCb_t                                     pfnEvictImageCb;
} ze_context_callbacks_t;

typedef struct _ze_command_queue_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_command_queue_desc_t** pdesc;
ze_command_queue_handle_t** pphCommandQueue;
} ze_command_queue_create_params_t;

typedef void (ZE_APICALL *ze_pfnCommandQueueCreateCb_t)(
ze_command_queue_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_queue_destroy_params_t
{
ze_command_queue_handle_t* phCommandQueue;
} ze_command_queue_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnCommandQueueDestroyCb_t)(
ze_command_queue_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_queue_execute_command_lists_params_t
{
ze_command_queue_handle_t* phCommandQueue;
uint32_t* pnumCommandLists;
ze_command_list_handle_t** pphCommandLists;
ze_fence_handle_t* phFence;
} ze_command_queue_execute_command_lists_params_t;

typedef void (ZE_APICALL *ze_pfnCommandQueueExecuteCommandListsCb_t)(
ze_command_queue_execute_command_lists_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_queue_synchronize_params_t
{
ze_command_queue_handle_t* phCommandQueue;
uint64_t* ptimeout;
} ze_command_queue_synchronize_params_t;

typedef void (ZE_APICALL *ze_pfnCommandQueueSynchronizeCb_t)(
ze_command_queue_synchronize_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_queue_callbacks_t
{
ze_pfnCommandQueueCreateCb_t                                    pfnCreateCb;
ze_pfnCommandQueueDestroyCb_t                                   pfnDestroyCb;
ze_pfnCommandQueueExecuteCommandListsCb_t                       pfnExecuteCommandListsCb;
ze_pfnCommandQueueSynchronizeCb_t                               pfnSynchronizeCb;
} ze_command_queue_callbacks_t;

typedef struct _ze_command_list_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_command_list_desc_t** pdesc;
ze_command_list_handle_t** pphCommandList;
} ze_command_list_create_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListCreateCb_t)(
ze_command_list_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_create_immediate_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_command_queue_desc_t** paltdesc;
ze_command_list_handle_t** pphCommandList;
} ze_command_list_create_immediate_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListCreateImmediateCb_t)(
ze_command_list_create_immediate_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_destroy_params_t
{
ze_command_list_handle_t* phCommandList;
} ze_command_list_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListDestroyCb_t)(
ze_command_list_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_close_params_t
{
ze_command_list_handle_t* phCommandList;
} ze_command_list_close_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListCloseCb_t)(
ze_command_list_close_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_reset_params_t
{
ze_command_list_handle_t* phCommandList;
} ze_command_list_reset_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListResetCb_t)(
ze_command_list_reset_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_write_global_timestamp_params_t
{
ze_command_list_handle_t* phCommandList;
uint64_t** pdstptr;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_write_global_timestamp_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendWriteGlobalTimestampCb_t)(
ze_command_list_append_write_global_timestamp_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_barrier_params_t
{
ze_command_list_handle_t* phCommandList;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_barrier_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendBarrierCb_t)(
ze_command_list_append_barrier_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_ranges_barrier_params_t
{
ze_command_list_handle_t* phCommandList;
uint32_t* pnumRanges;
const size_t** ppRangeSizes;
const void*** ppRanges;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_ranges_barrier_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryRangesBarrierCb_t)(
ze_command_list_append_memory_ranges_barrier_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_copy_params_t
{
ze_command_list_handle_t* phCommandList;
void** pdstptr;
const void** psrcptr;
size_t* psize;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyCb_t)(
ze_command_list_append_memory_copy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_fill_params_t
{
ze_command_list_handle_t* phCommandList;
void** pptr;
const void** ppattern;
size_t* ppattern_size;
size_t* psize;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_fill_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryFillCb_t)(
ze_command_list_append_memory_fill_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_copy_region_params_t
{
ze_command_list_handle_t* phCommandList;
void** pdstptr;
const ze_copy_region_t** pdstRegion;
uint32_t* pdstPitch;
uint32_t* pdstSlicePitch;
const void** psrcptr;
const ze_copy_region_t** psrcRegion;
uint32_t* psrcPitch;
uint32_t* psrcSlicePitch;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_region_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyRegionCb_t)(
ze_command_list_append_memory_copy_region_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_copy_from_context_params_t
{
ze_command_list_handle_t* phCommandList;
void** pdstptr;
ze_context_handle_t* phContextSrc;
const void** psrcptr;
size_t* psize;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_memory_copy_from_context_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyFromContextCb_t)(
ze_command_list_append_memory_copy_from_context_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_image_copy_params_t
{
ze_command_list_handle_t* phCommandList;
ze_image_handle_t* phDstImage;
ze_image_handle_t* phSrcImage;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyCb_t)(
ze_command_list_append_image_copy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_image_copy_region_params_t
{
ze_command_list_handle_t* phCommandList;
ze_image_handle_t* phDstImage;
ze_image_handle_t* phSrcImage;
const ze_image_region_t** ppDstRegion;
const ze_image_region_t** ppSrcRegion;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_region_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyRegionCb_t)(
ze_command_list_append_image_copy_region_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_image_copy_to_memory_params_t
{
ze_command_list_handle_t* phCommandList;
void** pdstptr;
ze_image_handle_t* phSrcImage;
const ze_image_region_t** ppSrcRegion;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_to_memory_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyToMemoryCb_t)(
ze_command_list_append_image_copy_to_memory_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_image_copy_from_memory_params_t
{
ze_command_list_handle_t* phCommandList;
ze_image_handle_t* phDstImage;
const void** psrcptr;
const ze_image_region_t** ppDstRegion;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_image_copy_from_memory_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendImageCopyFromMemoryCb_t)(
ze_command_list_append_image_copy_from_memory_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_memory_prefetch_params_t
{
ze_command_list_handle_t* phCommandList;
const void** pptr;
size_t* psize;
} ze_command_list_append_memory_prefetch_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemoryPrefetchCb_t)(
ze_command_list_append_memory_prefetch_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_mem_advise_params_t
{
ze_command_list_handle_t* phCommandList;
ze_device_handle_t* phDevice;
const void** pptr;
size_t* psize;
ze_memory_advice_t* padvice;
} ze_command_list_append_mem_advise_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendMemAdviseCb_t)(
ze_command_list_append_mem_advise_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_signal_event_params_t
{
ze_command_list_handle_t* phCommandList;
ze_event_handle_t* phEvent;
} ze_command_list_append_signal_event_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendSignalEventCb_t)(
ze_command_list_append_signal_event_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_wait_on_events_params_t
{
ze_command_list_handle_t* phCommandList;
uint32_t* pnumEvents;
ze_event_handle_t** pphEvents;
} ze_command_list_append_wait_on_events_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendWaitOnEventsCb_t)(
ze_command_list_append_wait_on_events_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_event_reset_params_t
{
ze_command_list_handle_t* phCommandList;
ze_event_handle_t* phEvent;
} ze_command_list_append_event_reset_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendEventResetCb_t)(
ze_command_list_append_event_reset_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_query_kernel_timestamps_params_t
{
ze_command_list_handle_t* phCommandList;
uint32_t* pnumEvents;
ze_event_handle_t** pphEvents;
void** pdstptr;
const size_t** ppOffsets;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_query_kernel_timestamps_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendQueryKernelTimestampsCb_t)(
ze_command_list_append_query_kernel_timestamps_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_launch_kernel_params_t
{
ze_command_list_handle_t* phCommandList;
ze_kernel_handle_t* phKernel;
const ze_group_count_t** ppLaunchFuncArgs;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_kernel_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchKernelCb_t)(
ze_command_list_append_launch_kernel_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_launch_cooperative_kernel_params_t
{
ze_command_list_handle_t* phCommandList;
ze_kernel_handle_t* phKernel;
const ze_group_count_t** ppLaunchFuncArgs;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_cooperative_kernel_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchCooperativeKernelCb_t)(
ze_command_list_append_launch_cooperative_kernel_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_launch_kernel_indirect_params_t
{
ze_command_list_handle_t* phCommandList;
ze_kernel_handle_t* phKernel;
const ze_group_count_t** ppLaunchArgumentsBuffer;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_kernel_indirect_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchKernelIndirectCb_t)(
ze_command_list_append_launch_kernel_indirect_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_append_launch_multiple_kernels_indirect_params_t
{
ze_command_list_handle_t* phCommandList;
uint32_t* pnumKernels;
ze_kernel_handle_t** pphKernels;
const uint32_t** ppCountBuffer;
const ze_group_count_t** ppLaunchArgumentsBuffer;
ze_event_handle_t* phSignalEvent;
uint32_t* pnumWaitEvents;
ze_event_handle_t** pphWaitEvents;
} ze_command_list_append_launch_multiple_kernels_indirect_params_t;

typedef void (ZE_APICALL *ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t)(
ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_command_list_callbacks_t
{
ze_pfnCommandListCreateCb_t                                     pfnCreateCb;
ze_pfnCommandListCreateImmediateCb_t                            pfnCreateImmediateCb;
ze_pfnCommandListDestroyCb_t                                    pfnDestroyCb;
ze_pfnCommandListCloseCb_t                                      pfnCloseCb;
ze_pfnCommandListResetCb_t                                      pfnResetCb;
ze_pfnCommandListAppendWriteGlobalTimestampCb_t                 pfnAppendWriteGlobalTimestampCb;
ze_pfnCommandListAppendBarrierCb_t                              pfnAppendBarrierCb;
ze_pfnCommandListAppendMemoryRangesBarrierCb_t                  pfnAppendMemoryRangesBarrierCb;
ze_pfnCommandListAppendMemoryCopyCb_t                           pfnAppendMemoryCopyCb;
ze_pfnCommandListAppendMemoryFillCb_t                           pfnAppendMemoryFillCb;
ze_pfnCommandListAppendMemoryCopyRegionCb_t                     pfnAppendMemoryCopyRegionCb;
ze_pfnCommandListAppendMemoryCopyFromContextCb_t                pfnAppendMemoryCopyFromContextCb;
ze_pfnCommandListAppendImageCopyCb_t                            pfnAppendImageCopyCb;
ze_pfnCommandListAppendImageCopyRegionCb_t                      pfnAppendImageCopyRegionCb;
ze_pfnCommandListAppendImageCopyToMemoryCb_t                    pfnAppendImageCopyToMemoryCb;
ze_pfnCommandListAppendImageCopyFromMemoryCb_t                  pfnAppendImageCopyFromMemoryCb;
ze_pfnCommandListAppendMemoryPrefetchCb_t                       pfnAppendMemoryPrefetchCb;
ze_pfnCommandListAppendMemAdviseCb_t                            pfnAppendMemAdviseCb;
ze_pfnCommandListAppendSignalEventCb_t                          pfnAppendSignalEventCb;
ze_pfnCommandListAppendWaitOnEventsCb_t                         pfnAppendWaitOnEventsCb;
ze_pfnCommandListAppendEventResetCb_t                           pfnAppendEventResetCb;
ze_pfnCommandListAppendQueryKernelTimestampsCb_t                pfnAppendQueryKernelTimestampsCb;
ze_pfnCommandListAppendLaunchKernelCb_t                         pfnAppendLaunchKernelCb;
ze_pfnCommandListAppendLaunchCooperativeKernelCb_t              pfnAppendLaunchCooperativeKernelCb;
ze_pfnCommandListAppendLaunchKernelIndirectCb_t                 pfnAppendLaunchKernelIndirectCb;
ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t        pfnAppendLaunchMultipleKernelsIndirectCb;
} ze_command_list_callbacks_t;

typedef struct _ze_image_get_properties_params_t
{
ze_device_handle_t* phDevice;
const ze_image_desc_t** pdesc;
ze_image_properties_t** ppImageProperties;
} ze_image_get_properties_params_t;

typedef void (ZE_APICALL *ze_pfnImageGetPropertiesCb_t)(
ze_image_get_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_image_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_image_desc_t** pdesc;
ze_image_handle_t** pphImage;
} ze_image_create_params_t;

typedef void (ZE_APICALL *ze_pfnImageCreateCb_t)(
ze_image_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_image_destroy_params_t
{
ze_image_handle_t* phImage;
} ze_image_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnImageDestroyCb_t)(
ze_image_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_image_callbacks_t
{
ze_pfnImageGetPropertiesCb_t                                    pfnGetPropertiesCb;
ze_pfnImageCreateCb_t                                           pfnCreateCb;
ze_pfnImageDestroyCb_t                                          pfnDestroyCb;
} ze_image_callbacks_t;

typedef struct _ze_fence_create_params_t
{
ze_command_queue_handle_t* phCommandQueue;
const ze_fence_desc_t** pdesc;
ze_fence_handle_t** pphFence;
} ze_fence_create_params_t;

typedef void (ZE_APICALL *ze_pfnFenceCreateCb_t)(
ze_fence_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_fence_destroy_params_t
{
ze_fence_handle_t* phFence;
} ze_fence_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnFenceDestroyCb_t)(
ze_fence_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_fence_host_synchronize_params_t
{
ze_fence_handle_t* phFence;
uint64_t* ptimeout;
} ze_fence_host_synchronize_params_t;

typedef void (ZE_APICALL *ze_pfnFenceHostSynchronizeCb_t)(
ze_fence_host_synchronize_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_fence_query_status_params_t
{
ze_fence_handle_t* phFence;
} ze_fence_query_status_params_t;

typedef void (ZE_APICALL *ze_pfnFenceQueryStatusCb_t)(
ze_fence_query_status_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_fence_reset_params_t
{
ze_fence_handle_t* phFence;
} ze_fence_reset_params_t;

typedef void (ZE_APICALL *ze_pfnFenceResetCb_t)(
ze_fence_reset_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_fence_callbacks_t
{
ze_pfnFenceCreateCb_t                                           pfnCreateCb;
ze_pfnFenceDestroyCb_t                                          pfnDestroyCb;
ze_pfnFenceHostSynchronizeCb_t                                  pfnHostSynchronizeCb;
ze_pfnFenceQueryStatusCb_t                                      pfnQueryStatusCb;
ze_pfnFenceResetCb_t                                            pfnResetCb;
} ze_fence_callbacks_t;

typedef struct _ze_event_pool_create_params_t
{
ze_context_handle_t* phContext;
const ze_event_pool_desc_t** pdesc;
uint32_t* pnumDevices;
ze_device_handle_t** pphDevices;
ze_event_pool_handle_t** pphEventPool;
} ze_event_pool_create_params_t;

typedef void (ZE_APICALL *ze_pfnEventPoolCreateCb_t)(
ze_event_pool_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_pool_destroy_params_t
{
ze_event_pool_handle_t* phEventPool;
} ze_event_pool_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnEventPoolDestroyCb_t)(
ze_event_pool_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_pool_get_ipc_handle_params_t
{
ze_event_pool_handle_t* phEventPool;
ze_ipc_event_pool_handle_t** pphIpc;
} ze_event_pool_get_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnEventPoolGetIpcHandleCb_t)(
ze_event_pool_get_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_pool_open_ipc_handle_params_t
{
ze_context_handle_t* phContext;
ze_ipc_event_pool_handle_t* phIpc;
ze_event_pool_handle_t** pphEventPool;
} ze_event_pool_open_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnEventPoolOpenIpcHandleCb_t)(
ze_event_pool_open_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_pool_close_ipc_handle_params_t
{
ze_event_pool_handle_t* phEventPool;
} ze_event_pool_close_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnEventPoolCloseIpcHandleCb_t)(
ze_event_pool_close_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_pool_callbacks_t
{
ze_pfnEventPoolCreateCb_t                                       pfnCreateCb;
ze_pfnEventPoolDestroyCb_t                                      pfnDestroyCb;
ze_pfnEventPoolGetIpcHandleCb_t                                 pfnGetIpcHandleCb;
ze_pfnEventPoolOpenIpcHandleCb_t                                pfnOpenIpcHandleCb;
ze_pfnEventPoolCloseIpcHandleCb_t                               pfnCloseIpcHandleCb;
} ze_event_pool_callbacks_t;

typedef struct _ze_event_create_params_t
{
ze_event_pool_handle_t* phEventPool;
const ze_event_desc_t** pdesc;
ze_event_handle_t** pphEvent;
} ze_event_create_params_t;

typedef void (ZE_APICALL *ze_pfnEventCreateCb_t)(
ze_event_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_destroy_params_t
{
ze_event_handle_t* phEvent;
} ze_event_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnEventDestroyCb_t)(
ze_event_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_host_signal_params_t
{
ze_event_handle_t* phEvent;
} ze_event_host_signal_params_t;

typedef void (ZE_APICALL *ze_pfnEventHostSignalCb_t)(
ze_event_host_signal_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_host_synchronize_params_t
{
ze_event_handle_t* phEvent;
uint64_t* ptimeout;
} ze_event_host_synchronize_params_t;

typedef void (ZE_APICALL *ze_pfnEventHostSynchronizeCb_t)(
ze_event_host_synchronize_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_query_status_params_t
{
ze_event_handle_t* phEvent;
} ze_event_query_status_params_t;

typedef void (ZE_APICALL *ze_pfnEventQueryStatusCb_t)(
ze_event_query_status_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_host_reset_params_t
{
ze_event_handle_t* phEvent;
} ze_event_host_reset_params_t;

typedef void (ZE_APICALL *ze_pfnEventHostResetCb_t)(
ze_event_host_reset_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_query_kernel_timestamp_params_t
{
ze_event_handle_t* phEvent;
ze_kernel_timestamp_result_t** pdstptr;
} ze_event_query_kernel_timestamp_params_t;

typedef void (ZE_APICALL *ze_pfnEventQueryKernelTimestampCb_t)(
ze_event_query_kernel_timestamp_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_event_callbacks_t
{
ze_pfnEventCreateCb_t                                           pfnCreateCb;
ze_pfnEventDestroyCb_t                                          pfnDestroyCb;
ze_pfnEventHostSignalCb_t                                       pfnHostSignalCb;
ze_pfnEventHostSynchronizeCb_t                                  pfnHostSynchronizeCb;
ze_pfnEventQueryStatusCb_t                                      pfnQueryStatusCb;
ze_pfnEventHostResetCb_t                                        pfnHostResetCb;
ze_pfnEventQueryKernelTimestampCb_t                             pfnQueryKernelTimestampCb;
} ze_event_callbacks_t;

typedef struct _ze_module_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_module_desc_t** pdesc;
ze_module_handle_t** pphModule;
ze_module_build_log_handle_t** pphBuildLog;
} ze_module_create_params_t;

typedef void (ZE_APICALL *ze_pfnModuleCreateCb_t)(
ze_module_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_destroy_params_t
{
ze_module_handle_t* phModule;
} ze_module_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnModuleDestroyCb_t)(
ze_module_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_dynamic_link_params_t
{
uint32_t* pnumModules;
ze_module_handle_t** pphModules;
ze_module_build_log_handle_t** pphLinkLog;
} ze_module_dynamic_link_params_t;

typedef void (ZE_APICALL *ze_pfnModuleDynamicLinkCb_t)(
ze_module_dynamic_link_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_get_native_binary_params_t
{
ze_module_handle_t* phModule;
size_t** ppSize;
uint8_t** ppModuleNativeBinary;
} ze_module_get_native_binary_params_t;

typedef void (ZE_APICALL *ze_pfnModuleGetNativeBinaryCb_t)(
ze_module_get_native_binary_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_get_global_pointer_params_t
{
ze_module_handle_t* phModule;
const char** ppGlobalName;
size_t** ppSize;
void*** ppptr;
} ze_module_get_global_pointer_params_t;

typedef void (ZE_APICALL *ze_pfnModuleGetGlobalPointerCb_t)(
ze_module_get_global_pointer_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_get_kernel_names_params_t
{
ze_module_handle_t* phModule;
uint32_t** ppCount;
const char*** ppNames;
} ze_module_get_kernel_names_params_t;

typedef void (ZE_APICALL *ze_pfnModuleGetKernelNamesCb_t)(
ze_module_get_kernel_names_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_get_properties_params_t
{
ze_module_handle_t* phModule;
ze_module_properties_t** ppModuleProperties;
} ze_module_get_properties_params_t;

typedef void (ZE_APICALL *ze_pfnModuleGetPropertiesCb_t)(
ze_module_get_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_get_function_pointer_params_t
{
ze_module_handle_t* phModule;
const char** ppFunctionName;
void*** ppfnFunction;
} ze_module_get_function_pointer_params_t;

typedef void (ZE_APICALL *ze_pfnModuleGetFunctionPointerCb_t)(
ze_module_get_function_pointer_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_callbacks_t
{
ze_pfnModuleCreateCb_t                                          pfnCreateCb;
ze_pfnModuleDestroyCb_t                                         pfnDestroyCb;
ze_pfnModuleDynamicLinkCb_t                                     pfnDynamicLinkCb;
ze_pfnModuleGetNativeBinaryCb_t                                 pfnGetNativeBinaryCb;
ze_pfnModuleGetGlobalPointerCb_t                                pfnGetGlobalPointerCb;
ze_pfnModuleGetKernelNamesCb_t                                  pfnGetKernelNamesCb;
ze_pfnModuleGetPropertiesCb_t                                   pfnGetPropertiesCb;
ze_pfnModuleGetFunctionPointerCb_t                              pfnGetFunctionPointerCb;
} ze_module_callbacks_t;

typedef struct _ze_module_build_log_destroy_params_t
{
ze_module_build_log_handle_t* phModuleBuildLog;
} ze_module_build_log_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnModuleBuildLogDestroyCb_t)(
ze_module_build_log_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_build_log_get_string_params_t
{
ze_module_build_log_handle_t* phModuleBuildLog;
size_t** ppSize;
char** ppBuildLog;
} ze_module_build_log_get_string_params_t;

typedef void (ZE_APICALL *ze_pfnModuleBuildLogGetStringCb_t)(
ze_module_build_log_get_string_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_module_build_log_callbacks_t
{
ze_pfnModuleBuildLogDestroyCb_t                                 pfnDestroyCb;
ze_pfnModuleBuildLogGetStringCb_t                               pfnGetStringCb;
} ze_module_build_log_callbacks_t;

typedef struct _ze_kernel_create_params_t
{
ze_module_handle_t* phModule;
const ze_kernel_desc_t** pdesc;
ze_kernel_handle_t** pphKernel;
} ze_kernel_create_params_t;

typedef void (ZE_APICALL *ze_pfnKernelCreateCb_t)(
ze_kernel_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_destroy_params_t
{
ze_kernel_handle_t* phKernel;
} ze_kernel_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnKernelDestroyCb_t)(
ze_kernel_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_set_cache_config_params_t
{
ze_kernel_handle_t* phKernel;
ze_cache_config_flags_t* pflags;
} ze_kernel_set_cache_config_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSetCacheConfigCb_t)(
ze_kernel_set_cache_config_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_set_group_size_params_t
{
ze_kernel_handle_t* phKernel;
uint32_t* pgroupSizeX;
uint32_t* pgroupSizeY;
uint32_t* pgroupSizeZ;
} ze_kernel_set_group_size_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSetGroupSizeCb_t)(
ze_kernel_set_group_size_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_suggest_group_size_params_t
{
ze_kernel_handle_t* phKernel;
uint32_t* pglobalSizeX;
uint32_t* pglobalSizeY;
uint32_t* pglobalSizeZ;
uint32_t** pgroupSizeX;
uint32_t** pgroupSizeY;
uint32_t** pgroupSizeZ;
} ze_kernel_suggest_group_size_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSuggestGroupSizeCb_t)(
ze_kernel_suggest_group_size_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_suggest_max_cooperative_group_count_params_t
{
ze_kernel_handle_t* phKernel;
uint32_t** ptotalGroupCount;
} ze_kernel_suggest_max_cooperative_group_count_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t)(
ze_kernel_suggest_max_cooperative_group_count_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_set_argument_value_params_t
{
ze_kernel_handle_t* phKernel;
uint32_t* pargIndex;
size_t* pargSize;
const void** ppArgValue;
} ze_kernel_set_argument_value_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSetArgumentValueCb_t)(
ze_kernel_set_argument_value_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_set_indirect_access_params_t
{
ze_kernel_handle_t* phKernel;
ze_kernel_indirect_access_flags_t* pflags;
} ze_kernel_set_indirect_access_params_t;

typedef void (ZE_APICALL *ze_pfnKernelSetIndirectAccessCb_t)(
ze_kernel_set_indirect_access_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_get_indirect_access_params_t
{
ze_kernel_handle_t* phKernel;
ze_kernel_indirect_access_flags_t** ppFlags;
} ze_kernel_get_indirect_access_params_t;

typedef void (ZE_APICALL *ze_pfnKernelGetIndirectAccessCb_t)(
ze_kernel_get_indirect_access_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_get_source_attributes_params_t
{
ze_kernel_handle_t* phKernel;
uint32_t** ppSize;
char*** ppString;
} ze_kernel_get_source_attributes_params_t;

typedef void (ZE_APICALL *ze_pfnKernelGetSourceAttributesCb_t)(
ze_kernel_get_source_attributes_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_get_properties_params_t
{
ze_kernel_handle_t* phKernel;
ze_kernel_properties_t** ppKernelProperties;
} ze_kernel_get_properties_params_t;

typedef void (ZE_APICALL *ze_pfnKernelGetPropertiesCb_t)(
ze_kernel_get_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_get_name_params_t
{
ze_kernel_handle_t* phKernel;
size_t** ppSize;
char** ppName;
} ze_kernel_get_name_params_t;

typedef void (ZE_APICALL *ze_pfnKernelGetNameCb_t)(
ze_kernel_get_name_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_kernel_callbacks_t
{
ze_pfnKernelCreateCb_t                                          pfnCreateCb;
ze_pfnKernelDestroyCb_t                                         pfnDestroyCb;
ze_pfnKernelSetCacheConfigCb_t                                  pfnSetCacheConfigCb;
ze_pfnKernelSetGroupSizeCb_t                                    pfnSetGroupSizeCb;
ze_pfnKernelSuggestGroupSizeCb_t                                pfnSuggestGroupSizeCb;
ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t                 pfnSuggestMaxCooperativeGroupCountCb;
ze_pfnKernelSetArgumentValueCb_t                                pfnSetArgumentValueCb;
ze_pfnKernelSetIndirectAccessCb_t                               pfnSetIndirectAccessCb;
ze_pfnKernelGetIndirectAccessCb_t                               pfnGetIndirectAccessCb;
ze_pfnKernelGetSourceAttributesCb_t                             pfnGetSourceAttributesCb;
ze_pfnKernelGetPropertiesCb_t                                   pfnGetPropertiesCb;
ze_pfnKernelGetNameCb_t                                         pfnGetNameCb;
} ze_kernel_callbacks_t;

typedef struct _ze_sampler_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
const ze_sampler_desc_t** pdesc;
ze_sampler_handle_t** pphSampler;
} ze_sampler_create_params_t;

typedef void (ZE_APICALL *ze_pfnSamplerCreateCb_t)(
ze_sampler_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_sampler_destroy_params_t
{
ze_sampler_handle_t* phSampler;
} ze_sampler_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnSamplerDestroyCb_t)(
ze_sampler_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_sampler_callbacks_t
{
ze_pfnSamplerCreateCb_t                                         pfnCreateCb;
ze_pfnSamplerDestroyCb_t                                        pfnDestroyCb;
} ze_sampler_callbacks_t;

typedef struct _ze_physical_mem_create_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
ze_physical_mem_desc_t** pdesc;
ze_physical_mem_handle_t** pphPhysicalMemory;
} ze_physical_mem_create_params_t;

typedef void (ZE_APICALL *ze_pfnPhysicalMemCreateCb_t)(
ze_physical_mem_create_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_physical_mem_destroy_params_t
{
ze_context_handle_t* phContext;
ze_physical_mem_handle_t* phPhysicalMemory;
} ze_physical_mem_destroy_params_t;

typedef void (ZE_APICALL *ze_pfnPhysicalMemDestroyCb_t)(
ze_physical_mem_destroy_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_physical_mem_callbacks_t
{
ze_pfnPhysicalMemCreateCb_t                                     pfnCreateCb;
ze_pfnPhysicalMemDestroyCb_t                                    pfnDestroyCb;
} ze_physical_mem_callbacks_t;

typedef struct _ze_mem_alloc_shared_params_t
{
ze_context_handle_t* phContext;
const ze_device_mem_alloc_desc_t** pdevice_desc;
const ze_host_mem_alloc_desc_t** phost_desc;
size_t* psize;
size_t* palignment;
ze_device_handle_t* phDevice;
void*** ppptr;
} ze_mem_alloc_shared_params_t;

typedef void (ZE_APICALL *ze_pfnMemAllocSharedCb_t)(
ze_mem_alloc_shared_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_alloc_device_params_t
{
ze_context_handle_t* phContext;
const ze_device_mem_alloc_desc_t** pdevice_desc;
size_t* psize;
size_t* palignment;
ze_device_handle_t* phDevice;
void*** ppptr;
} ze_mem_alloc_device_params_t;

typedef void (ZE_APICALL *ze_pfnMemAllocDeviceCb_t)(
ze_mem_alloc_device_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_alloc_host_params_t
{
ze_context_handle_t* phContext;
const ze_host_mem_alloc_desc_t** phost_desc;
size_t* psize;
size_t* palignment;
void*** ppptr;
} ze_mem_alloc_host_params_t;

typedef void (ZE_APICALL *ze_pfnMemAllocHostCb_t)(
ze_mem_alloc_host_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_free_params_t
{
ze_context_handle_t* phContext;
void** pptr;
} ze_mem_free_params_t;

typedef void (ZE_APICALL *ze_pfnMemFreeCb_t)(
ze_mem_free_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_get_alloc_properties_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
ze_memory_allocation_properties_t** ppMemAllocProperties;
ze_device_handle_t** pphDevice;
} ze_mem_get_alloc_properties_params_t;

typedef void (ZE_APICALL *ze_pfnMemGetAllocPropertiesCb_t)(
ze_mem_get_alloc_properties_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_get_address_range_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
void*** ppBase;
size_t** ppSize;
} ze_mem_get_address_range_params_t;

typedef void (ZE_APICALL *ze_pfnMemGetAddressRangeCb_t)(
ze_mem_get_address_range_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_get_ipc_handle_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
ze_ipc_mem_handle_t** ppIpcHandle;
} ze_mem_get_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnMemGetIpcHandleCb_t)(
ze_mem_get_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_open_ipc_handle_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
ze_ipc_mem_handle_t* phandle;
ze_ipc_memory_flags_t* pflags;
void*** ppptr;
} ze_mem_open_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnMemOpenIpcHandleCb_t)(
ze_mem_open_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_close_ipc_handle_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
} ze_mem_close_ipc_handle_params_t;

typedef void (ZE_APICALL *ze_pfnMemCloseIpcHandleCb_t)(
ze_mem_close_ipc_handle_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_mem_callbacks_t
{
ze_pfnMemAllocSharedCb_t                                        pfnAllocSharedCb;
ze_pfnMemAllocDeviceCb_t                                        pfnAllocDeviceCb;
ze_pfnMemAllocHostCb_t                                          pfnAllocHostCb;
ze_pfnMemFreeCb_t                                               pfnFreeCb;
ze_pfnMemGetAllocPropertiesCb_t                                 pfnGetAllocPropertiesCb;
ze_pfnMemGetAddressRangeCb_t                                    pfnGetAddressRangeCb;
ze_pfnMemGetIpcHandleCb_t                                       pfnGetIpcHandleCb;
ze_pfnMemOpenIpcHandleCb_t                                      pfnOpenIpcHandleCb;
ze_pfnMemCloseIpcHandleCb_t                                     pfnCloseIpcHandleCb;
} ze_mem_callbacks_t;

typedef struct _ze_virtual_mem_reserve_params_t
{
ze_context_handle_t* phContext;
const void** ppStart;
size_t* psize;
void*** ppptr;
} ze_virtual_mem_reserve_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemReserveCb_t)(
ze_virtual_mem_reserve_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_free_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
size_t* psize;
} ze_virtual_mem_free_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemFreeCb_t)(
ze_virtual_mem_free_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_query_page_size_params_t
{
ze_context_handle_t* phContext;
ze_device_handle_t* phDevice;
size_t* psize;
size_t** ppagesize;
} ze_virtual_mem_query_page_size_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemQueryPageSizeCb_t)(
ze_virtual_mem_query_page_size_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_map_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
size_t* psize;
ze_physical_mem_handle_t* phPhysicalMemory;
size_t* poffset;
ze_memory_access_attribute_t* paccess;
} ze_virtual_mem_map_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemMapCb_t)(
ze_virtual_mem_map_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_unmap_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
size_t* psize;
} ze_virtual_mem_unmap_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemUnmapCb_t)(
ze_virtual_mem_unmap_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_set_access_attribute_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
size_t* psize;
ze_memory_access_attribute_t* paccess;
} ze_virtual_mem_set_access_attribute_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemSetAccessAttributeCb_t)(
ze_virtual_mem_set_access_attribute_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_get_access_attribute_params_t
{
ze_context_handle_t* phContext;
const void** pptr;
size_t* psize;
ze_memory_access_attribute_t** paccess;
size_t** poutSize;
} ze_virtual_mem_get_access_attribute_params_t;

typedef void (ZE_APICALL *ze_pfnVirtualMemGetAccessAttributeCb_t)(
ze_virtual_mem_get_access_attribute_params_t* params,
ze_result_t result,
void* pTracerUserData,
void** ppTracerInstanceUserData
);

typedef struct _ze_virtual_mem_callbacks_t
{
ze_pfnVirtualMemReserveCb_t                                     pfnReserveCb;
ze_pfnVirtualMemFreeCb_t                                        pfnFreeCb;
ze_pfnVirtualMemQueryPageSizeCb_t                               pfnQueryPageSizeCb;
ze_pfnVirtualMemMapCb_t                                         pfnMapCb;
ze_pfnVirtualMemUnmapCb_t                                       pfnUnmapCb;
ze_pfnVirtualMemSetAccessAttributeCb_t                          pfnSetAccessAttributeCb;
ze_pfnVirtualMemGetAccessAttributeCb_t                          pfnGetAccessAttributeCb;
} ze_virtual_mem_callbacks_t;

typedef struct _ze_callbacks_t
{
ze_global_callbacks_t               Global;
ze_driver_callbacks_t               Driver;
ze_device_callbacks_t               Device;
ze_context_callbacks_t              Context;
ze_command_queue_callbacks_t        CommandQueue;
ze_command_list_callbacks_t         CommandList;
ze_fence_callbacks_t                Fence;
ze_event_pool_callbacks_t           EventPool;
ze_event_callbacks_t                Event;
ze_image_callbacks_t                Image;
ze_module_callbacks_t               Module;
ze_module_build_log_callbacks_t     ModuleBuildLog;
ze_kernel_callbacks_t               Kernel;
ze_sampler_callbacks_t              Sampler;
ze_physical_mem_callbacks_t         PhysicalMem;
ze_mem_callbacks_t                  Mem;
ze_virtual_mem_callbacks_t          VirtualMem;
} ze_callbacks_t;
#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} 
#endif

#endif 