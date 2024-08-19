
#ifndef _ZET_API_H
#define _ZET_API_H
#if defined(__cplusplus)
#pragma once
#endif

#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(__GNUC__)
#pragma region common
#endif
typedef ze_driver_handle_t zet_driver_handle_t;

typedef ze_device_handle_t zet_device_handle_t;

typedef ze_context_handle_t zet_context_handle_t;

typedef ze_command_list_handle_t zet_command_list_handle_t;

typedef ze_module_handle_t zet_module_handle_t;

typedef ze_kernel_handle_t zet_kernel_handle_t;

typedef struct _zet_metric_group_handle_t *zet_metric_group_handle_t;

typedef struct _zet_metric_handle_t *zet_metric_handle_t;

typedef struct _zet_metric_streamer_handle_t *zet_metric_streamer_handle_t;

typedef struct _zet_metric_query_pool_handle_t *zet_metric_query_pool_handle_t;

typedef struct _zet_metric_query_handle_t *zet_metric_query_handle_t;

typedef struct _zet_tracer_exp_handle_t *zet_tracer_exp_handle_t;

typedef struct _zet_debug_session_handle_t *zet_debug_session_handle_t;

typedef enum _zet_structure_type_t
{
ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES = 0x1,   
ZET_STRUCTURE_TYPE_METRIC_PROPERTIES = 0x2,     
ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC = 0x3,  
ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC = 0x4,
ZET_STRUCTURE_TYPE_PROFILE_PROPERTIES = 0x5,    
ZET_STRUCTURE_TYPE_DEVICE_DEBUG_PROPERTIES = 0x6,   
ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC = 0x7,   
ZET_STRUCTURE_TYPE_DEBUG_REGSET_PROPERTIES = 0x8,   
ZET_STRUCTURE_TYPE_TRACER_EXP_DESC = 0x00010001,
ZET_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_structure_type_t;

typedef struct _zet_base_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    

} zet_base_properties_t;

typedef struct _zet_base_desc_t
{
zet_structure_type_t stype;                     
const void* pNext;                              

} zet_base_desc_t;

typedef enum _zet_value_type_t
{
ZET_VALUE_TYPE_UINT32 = 0,                      
ZET_VALUE_TYPE_UINT64 = 1,                      
ZET_VALUE_TYPE_FLOAT32 = 2,                     
ZET_VALUE_TYPE_FLOAT64 = 3,                     
ZET_VALUE_TYPE_BOOL8 = 4,                       
ZET_VALUE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_value_type_t;

typedef union _zet_value_t
{
uint32_t ui32;                                  
uint64_t ui64;                                  
float fp32;                                     
double fp64;                                    
ze_bool_t b8;                                   

} zet_value_t;

typedef struct _zet_typed_value_t
{
zet_value_type_t type;                          
zet_value_t value;                              

} zet_typed_value_t;

typedef struct _zet_base_properties_t zet_base_properties_t;

typedef struct _zet_base_desc_t zet_base_desc_t;

typedef struct _zet_typed_value_t zet_typed_value_t;

typedef struct _zet_device_debug_properties_t zet_device_debug_properties_t;

typedef struct _zet_debug_config_t zet_debug_config_t;

typedef struct _zet_debug_event_info_detached_t zet_debug_event_info_detached_t;

typedef struct _zet_debug_event_info_module_t zet_debug_event_info_module_t;

typedef struct _zet_debug_event_info_thread_stopped_t zet_debug_event_info_thread_stopped_t;

typedef struct _zet_debug_event_info_page_fault_t zet_debug_event_info_page_fault_t;

typedef struct _zet_debug_event_t zet_debug_event_t;

typedef struct _zet_debug_memory_space_desc_t zet_debug_memory_space_desc_t;

typedef struct _zet_debug_regset_properties_t zet_debug_regset_properties_t;

typedef struct _zet_metric_group_properties_t zet_metric_group_properties_t;

typedef struct _zet_metric_properties_t zet_metric_properties_t;

typedef struct _zet_metric_streamer_desc_t zet_metric_streamer_desc_t;

typedef struct _zet_metric_query_pool_desc_t zet_metric_query_pool_desc_t;

typedef struct _zet_profile_properties_t zet_profile_properties_t;

typedef struct _zet_profile_free_register_token_t zet_profile_free_register_token_t;

typedef struct _zet_profile_register_sequence_t zet_profile_register_sequence_t;

typedef struct _zet_tracer_exp_desc_t zet_tracer_exp_desc_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region device
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region context
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region cmdlist
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region module
#endif
typedef enum _zet_module_debug_info_format_t
{
ZET_MODULE_DEBUG_INFO_FORMAT_ELF_DWARF = 0,     
ZET_MODULE_DEBUG_INFO_FORMAT_FORCE_UINT32 = 0x7fffffff

} zet_module_debug_info_format_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetModuleGetDebugInfo(
zet_module_handle_t hModule,                    
zet_module_debug_info_format_t format,          
size_t* pSize,                                  
uint8_t* pDebugInfo                             
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region debug
#endif
typedef uint32_t zet_device_debug_property_flags_t;
typedef enum _zet_device_debug_property_flag_t
{
ZET_DEVICE_DEBUG_PROPERTY_FLAG_ATTACH = ZE_BIT(0),  
ZET_DEVICE_DEBUG_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_device_debug_property_flag_t;

typedef struct _zet_device_debug_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    
zet_device_debug_property_flags_t flags;        

} zet_device_debug_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDeviceGetDebugProperties(
zet_device_handle_t hDevice,                    
zet_device_debug_properties_t* pDebugProperties 
);

typedef struct _zet_debug_config_t
{
uint32_t pid;                                   

} zet_debug_config_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugAttach(
zet_device_handle_t hDevice,                    
const zet_debug_config_t* config,               
zet_debug_session_handle_t* phDebug             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugDetach(
zet_debug_session_handle_t hDebug               
);

typedef uint32_t zet_debug_event_flags_t;
typedef enum _zet_debug_event_flag_t
{
ZET_DEBUG_EVENT_FLAG_NEED_ACK = ZE_BIT(0),      
ZET_DEBUG_EVENT_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_flag_t;

typedef enum _zet_debug_event_type_t
{
ZET_DEBUG_EVENT_TYPE_INVALID = 0,               
ZET_DEBUG_EVENT_TYPE_DETACHED = 1,              
ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY = 2,         
ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT = 3,          
ZET_DEBUG_EVENT_TYPE_MODULE_LOAD = 4,           
ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD = 5,         
ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED = 6,        
ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE = 7,    
ZET_DEBUG_EVENT_TYPE_PAGE_FAULT = 8,            
ZET_DEBUG_EVENT_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_type_t;

typedef enum _zet_debug_detach_reason_t
{
ZET_DEBUG_DETACH_REASON_INVALID = 0,            
ZET_DEBUG_DETACH_REASON_HOST_EXIT = 1,          
ZET_DEBUG_DETACH_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_detach_reason_t;

typedef struct _zet_debug_event_info_detached_t
{
zet_debug_detach_reason_t reason;               

} zet_debug_event_info_detached_t;

typedef struct _zet_debug_event_info_module_t
{
zet_module_debug_info_format_t format;          
uint64_t moduleBegin;                           
uint64_t moduleEnd;                             
uint64_t load;                                  

} zet_debug_event_info_module_t;

typedef struct _zet_debug_event_info_thread_stopped_t
{
ze_device_thread_t thread;                      

} zet_debug_event_info_thread_stopped_t;

typedef enum _zet_debug_page_fault_reason_t
{
ZET_DEBUG_PAGE_FAULT_REASON_INVALID = 0,        
ZET_DEBUG_PAGE_FAULT_REASON_MAPPING_ERROR = 1,  
ZET_DEBUG_PAGE_FAULT_REASON_PERMISSION_ERROR = 2,   
ZET_DEBUG_PAGE_FAULT_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_page_fault_reason_t;

typedef struct _zet_debug_event_info_page_fault_t
{
uint64_t address;                               
uint64_t mask;                                  
zet_debug_page_fault_reason_t reason;           

} zet_debug_event_info_page_fault_t;

typedef union _zet_debug_event_info_t
{
zet_debug_event_info_detached_t detached;       
zet_debug_event_info_module_t module;           
zet_debug_event_info_thread_stopped_t thread;   
zet_debug_event_info_page_fault_t page_fault;   

} zet_debug_event_info_t;

typedef struct _zet_debug_event_t
{
zet_debug_event_type_t type;                    
zet_debug_event_flags_t flags;                  
zet_debug_event_info_t info;                    

} zet_debug_event_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadEvent(
zet_debug_session_handle_t hDebug,              
uint64_t timeout,                               
zet_debug_event_t* event                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugAcknowledgeEvent(
zet_debug_session_handle_t hDebug,              
const zet_debug_event_t* event                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugInterrupt(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugResume(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread                       
);

typedef enum _zet_debug_memory_space_type_t
{
ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT = 0,        
ZET_DEBUG_MEMORY_SPACE_TYPE_SLM = 1,            
ZET_DEBUG_MEMORY_SPACE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_memory_space_type_t;

typedef struct _zet_debug_memory_space_desc_t
{
zet_structure_type_t stype;                     
const void* pNext;                              
zet_debug_memory_space_type_t type;             
uint64_t address;                               

} zet_debug_memory_space_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadMemory(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread,                      
const zet_debug_memory_space_desc_t* desc,      
size_t size,                                    
void* buffer                                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteMemory(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread,                      
const zet_debug_memory_space_desc_t* desc,      
size_t size,                                    
const void* buffer                              
);

typedef uint32_t zet_debug_regset_flags_t;
typedef enum _zet_debug_regset_flag_t
{
ZET_DEBUG_REGSET_FLAG_READABLE = ZE_BIT(0),     
ZET_DEBUG_REGSET_FLAG_WRITEABLE = ZE_BIT(1),    
ZET_DEBUG_REGSET_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_regset_flag_t;

typedef struct _zet_debug_regset_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    
uint32_t type;                                  
uint32_t version;                               
zet_debug_regset_flags_t generalFlags;          
uint32_t deviceFlags;                           
uint32_t count;                                 
uint32_t bitSize;                               
uint32_t byteSize;                              

} zet_debug_regset_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugGetRegisterSetProperties(
zet_device_handle_t hDevice,                    
uint32_t* pCount,                               
zet_debug_regset_properties_t* pRegisterSetProperties   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadRegisters(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread,                      
uint32_t type,                                  
uint32_t start,                                 
uint32_t count,                                 
void* pRegisterValues                           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteRegisters(
zet_debug_session_handle_t hDebug,              
ze_device_thread_t thread,                      
uint32_t type,                                  
uint32_t start,                                 
uint32_t count,                                 
void* pRegisterValues                           
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region metric
#endif
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGet(
zet_device_handle_t hDevice,                    
uint32_t* pCount,                               
zet_metric_group_handle_t* phMetricGroups       
);

#ifndef ZET_MAX_METRIC_GROUP_NAME
#define ZET_MAX_METRIC_GROUP_NAME  256
#endif 

#ifndef ZET_MAX_METRIC_GROUP_DESCRIPTION
#define ZET_MAX_METRIC_GROUP_DESCRIPTION  256
#endif 

typedef uint32_t zet_metric_group_sampling_type_flags_t;
typedef enum _zet_metric_group_sampling_type_flag_t
{
ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED = ZE_BIT(0),
ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED = ZE_BIT(1), 
ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_sampling_type_flag_t;

typedef struct _zet_metric_group_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    
char name[ZET_MAX_METRIC_GROUP_NAME];           
char description[ZET_MAX_METRIC_GROUP_DESCRIPTION]; 
zet_metric_group_sampling_type_flags_t samplingType;
uint32_t domain;                                
uint32_t metricCount;                           

} zet_metric_group_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGetProperties(
zet_metric_group_handle_t hMetricGroup,         
zet_metric_group_properties_t* pProperties      
);

typedef enum _zet_metric_type_t
{
ZET_METRIC_TYPE_DURATION = 0,                   
ZET_METRIC_TYPE_EVENT = 1,                      
ZET_METRIC_TYPE_EVENT_WITH_RANGE = 2,           
ZET_METRIC_TYPE_THROUGHPUT = 3,                 
ZET_METRIC_TYPE_TIMESTAMP = 4,                  
ZET_METRIC_TYPE_FLAG = 5,                       
ZET_METRIC_TYPE_RATIO = 6,                      
ZET_METRIC_TYPE_RAW = 7,                        
ZET_METRIC_TYPE_IP_EXP = 0x7ffffffe,            
ZET_METRIC_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_type_t;

typedef enum _zet_metric_group_calculation_type_t
{
ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES = 0,
ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES = 1,
ZET_METRIC_GROUP_CALCULATION_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_calculation_type_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMetricValues(
zet_metric_group_handle_t hMetricGroup,         
zet_metric_group_calculation_type_t type,       
size_t rawDataSize,                             
const uint8_t* pRawData,                        
uint32_t* pMetricValueCount,                    
zet_typed_value_t* pMetricValues                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGet(
zet_metric_group_handle_t hMetricGroup,         
uint32_t* pCount,                               
zet_metric_handle_t* phMetrics                  
);

#ifndef ZET_MAX_METRIC_NAME
#define ZET_MAX_METRIC_NAME  256
#endif 

#ifndef ZET_MAX_METRIC_DESCRIPTION
#define ZET_MAX_METRIC_DESCRIPTION  256
#endif 

#ifndef ZET_MAX_METRIC_COMPONENT
#define ZET_MAX_METRIC_COMPONENT  256
#endif 

#ifndef ZET_MAX_METRIC_RESULT_UNITS
#define ZET_MAX_METRIC_RESULT_UNITS  256
#endif 

typedef struct _zet_metric_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    
char name[ZET_MAX_METRIC_NAME];                 
char description[ZET_MAX_METRIC_DESCRIPTION];   
char component[ZET_MAX_METRIC_COMPONENT];       
uint32_t tierNumber;                            
zet_metric_type_t metricType;                   
zet_value_type_t resultType;                    
char resultUnits[ZET_MAX_METRIC_RESULT_UNITS];  

} zet_metric_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGetProperties(
zet_metric_handle_t hMetric,                    
zet_metric_properties_t* pProperties            
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetContextActivateMetricGroups(
zet_context_handle_t hContext,                  
zet_device_handle_t hDevice,                    
uint32_t count,                                 
zet_metric_group_handle_t* phMetricGroups       
);

typedef struct _zet_metric_streamer_desc_t
{
zet_structure_type_t stype;                     
const void* pNext;                              
uint32_t notifyEveryNReports;                   
uint32_t samplingPeriod;                        

} zet_metric_streamer_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerOpen(
zet_context_handle_t hContext,                  
zet_device_handle_t hDevice,                    
zet_metric_group_handle_t hMetricGroup,         
zet_metric_streamer_desc_t* desc,               
ze_event_handle_t hNotificationEvent,           
zet_metric_streamer_handle_t* phMetricStreamer  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricStreamerMarker(
zet_command_list_handle_t hCommandList,         
zet_metric_streamer_handle_t hMetricStreamer,   
uint32_t value                                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerClose(
zet_metric_streamer_handle_t hMetricStreamer    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerReadData(
zet_metric_streamer_handle_t hMetricStreamer,   
uint32_t maxReportCount,                        
size_t* pRawDataSize,                           
uint8_t* pRawData                               
);

typedef enum _zet_metric_query_pool_type_t
{
ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE = 0,     
ZET_METRIC_QUERY_POOL_TYPE_EXECUTION = 1,       
ZET_METRIC_QUERY_POOL_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_query_pool_type_t;

typedef struct _zet_metric_query_pool_desc_t
{
zet_structure_type_t stype;                     
const void* pNext;                              
zet_metric_query_pool_type_t type;              
uint32_t count;                                 

} zet_metric_query_pool_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryPoolCreate(
zet_context_handle_t hContext,                  
zet_device_handle_t hDevice,                    
zet_metric_group_handle_t hMetricGroup,         
const zet_metric_query_pool_desc_t* desc,       
zet_metric_query_pool_handle_t* phMetricQueryPool   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryPoolDestroy(
zet_metric_query_pool_handle_t hMetricQueryPool 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryCreate(
zet_metric_query_pool_handle_t hMetricQueryPool,
uint32_t index,                                 
zet_metric_query_handle_t* phMetricQuery        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryDestroy(
zet_metric_query_handle_t hMetricQuery          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryReset(
zet_metric_query_handle_t hMetricQuery          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryBegin(
zet_command_list_handle_t hCommandList,         
zet_metric_query_handle_t hMetricQuery          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryEnd(
zet_command_list_handle_t hCommandList,         
zet_metric_query_handle_t hMetricQuery,         
ze_event_handle_t hSignalEvent,                 
uint32_t numWaitEvents,                         
ze_event_handle_t* phWaitEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricMemoryBarrier(
zet_command_list_handle_t hCommandList          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryGetData(
zet_metric_query_handle_t hMetricQuery,         
size_t* pRawDataSize,                           
uint8_t* pRawData                               
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region pin
#endif
typedef uint32_t zet_profile_flags_t;
typedef enum _zet_profile_flag_t
{
ZET_PROFILE_FLAG_REGISTER_REALLOCATION = ZE_BIT(0), 
ZET_PROFILE_FLAG_FREE_REGISTER_INFO = ZE_BIT(1),
ZET_PROFILE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_profile_flag_t;

typedef struct _zet_profile_properties_t
{
zet_structure_type_t stype;                     
void* pNext;                                    
zet_profile_flags_t flags;                      
uint32_t numTokens;                             

} zet_profile_properties_t;

typedef enum _zet_profile_token_type_t
{
ZET_PROFILE_TOKEN_TYPE_FREE_REGISTER = 0,       
ZET_PROFILE_TOKEN_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_profile_token_type_t;

typedef struct _zet_profile_free_register_token_t
{
zet_profile_token_type_t type;                  
uint32_t size;                                  
uint32_t count;                                 

} zet_profile_free_register_token_t;

typedef struct _zet_profile_register_sequence_t
{
uint32_t start;                                 
uint32_t count;                                 

} zet_profile_register_sequence_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetKernelGetProfileInfo(
zet_kernel_handle_t hKernel,                    
zet_profile_properties_t* pProfileProperties    
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region tracing
#endif
#ifndef ZET_API_TRACING_EXP_NAME
#define ZET_API_TRACING_EXP_NAME  "ZET_experimental_api_tracing"
#endif 

typedef enum _zet_api_tracing_exp_version_t
{
ZET_API_TRACING_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZET_API_TRACING_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZET_API_TRACING_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_api_tracing_exp_version_t;

typedef ze_callbacks_t zet_core_callbacks_t;

typedef struct _zet_tracer_exp_desc_t
{
zet_structure_type_t stype;                     
const void* pNext;                              
void* pUserData;                                

} zet_tracer_exp_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpCreate(
zet_context_handle_t hContext,                  
const zet_tracer_exp_desc_t* desc,              
zet_tracer_exp_handle_t* phTracer               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpDestroy(
zet_tracer_exp_handle_t hTracer                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetPrologues(
zet_tracer_exp_handle_t hTracer,                
zet_core_callbacks_t* pCoreCbs                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetEpilogues(
zet_tracer_exp_handle_t hTracer,                
zet_core_callbacks_t* pCoreCbs                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetEnabled(
zet_tracer_exp_handle_t hTracer,                
ze_bool_t enable                                
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region multiMetricValues
#endif
#ifndef ZET_MULTI_METRICS_EXP_NAME
#define ZET_MULTI_METRICS_EXP_NAME  "ZET_experimental_calculate_multiple_metrics"
#endif 

typedef enum _ze_calculate_multiple_metrics_exp_version_t
{
ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),
ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),
ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_calculate_multiple_metrics_exp_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMultipleMetricValuesExp(
zet_metric_group_handle_t hMetricGroup,         
zet_metric_group_calculation_type_t type,       
size_t rawDataSize,                             
const uint8_t* pRawData,                        
uint32_t* pSetCount,                            
uint32_t* pTotalMetricValueCount,               
uint32_t* pMetricCounts,                        
zet_typed_value_t* pMetricValues                
);

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} 
#endif

#endif 