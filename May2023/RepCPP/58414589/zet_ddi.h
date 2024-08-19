
#ifndef _ZET_DDI_H
#define _ZET_DDI_H
#if defined(__cplusplus)
#pragma once
#endif
#include "zet_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef ze_result_t (ZE_APICALL *zet_pfnDeviceGetDebugProperties_t)(
zet_device_handle_t,
zet_device_debug_properties_t*
);

typedef struct _zet_device_dditable_t
{
zet_pfnDeviceGetDebugProperties_t                           pfnGetDebugProperties;
} zet_device_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetDeviceProcAddrTable(
ze_api_version_t version,                       
zet_device_dditable_t* pDdiTable                
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetDeviceProcAddrTable_t)(
ze_api_version_t,
zet_device_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnContextActivateMetricGroups_t)(
zet_context_handle_t,
zet_device_handle_t,
uint32_t,
zet_metric_group_handle_t*
);

typedef struct _zet_context_dditable_t
{
zet_pfnContextActivateMetricGroups_t                        pfnActivateMetricGroups;
} zet_context_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetContextProcAddrTable(
ze_api_version_t version,                       
zet_context_dditable_t* pDdiTable               
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetContextProcAddrTable_t)(
ze_api_version_t,
zet_context_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnCommandListAppendMetricStreamerMarker_t)(
zet_command_list_handle_t,
zet_metric_streamer_handle_t,
uint32_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnCommandListAppendMetricQueryBegin_t)(
zet_command_list_handle_t,
zet_metric_query_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnCommandListAppendMetricQueryEnd_t)(
zet_command_list_handle_t,
zet_metric_query_handle_t,
ze_event_handle_t,
uint32_t,
ze_event_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnCommandListAppendMetricMemoryBarrier_t)(
zet_command_list_handle_t
);

typedef struct _zet_command_list_dditable_t
{
zet_pfnCommandListAppendMetricStreamerMarker_t              pfnAppendMetricStreamerMarker;
zet_pfnCommandListAppendMetricQueryBegin_t                  pfnAppendMetricQueryBegin;
zet_pfnCommandListAppendMetricQueryEnd_t                    pfnAppendMetricQueryEnd;
zet_pfnCommandListAppendMetricMemoryBarrier_t               pfnAppendMetricMemoryBarrier;
} zet_command_list_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetCommandListProcAddrTable(
ze_api_version_t version,                       
zet_command_list_dditable_t* pDdiTable          
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetCommandListProcAddrTable_t)(
ze_api_version_t,
zet_command_list_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnModuleGetDebugInfo_t)(
zet_module_handle_t,
zet_module_debug_info_format_t,
size_t*,
uint8_t*
);

typedef struct _zet_module_dditable_t
{
zet_pfnModuleGetDebugInfo_t                                 pfnGetDebugInfo;
} zet_module_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetModuleProcAddrTable(
ze_api_version_t version,                       
zet_module_dditable_t* pDdiTable                
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetModuleProcAddrTable_t)(
ze_api_version_t,
zet_module_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnKernelGetProfileInfo_t)(
zet_kernel_handle_t,
zet_profile_properties_t*
);

typedef struct _zet_kernel_dditable_t
{
zet_pfnKernelGetProfileInfo_t                               pfnGetProfileInfo;
} zet_kernel_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetKernelProcAddrTable(
ze_api_version_t version,                       
zet_kernel_dditable_t* pDdiTable                
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetKernelProcAddrTable_t)(
ze_api_version_t,
zet_kernel_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGroupGet_t)(
zet_device_handle_t,
uint32_t*,
zet_metric_group_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGroupGetProperties_t)(
zet_metric_group_handle_t,
zet_metric_group_properties_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGroupCalculateMetricValues_t)(
zet_metric_group_handle_t,
zet_metric_group_calculation_type_t,
size_t,
const uint8_t*,
uint32_t*,
zet_typed_value_t*
);

typedef struct _zet_metric_group_dditable_t
{
zet_pfnMetricGroupGet_t                                     pfnGet;
zet_pfnMetricGroupGetProperties_t                           pfnGetProperties;
zet_pfnMetricGroupCalculateMetricValues_t                   pfnCalculateMetricValues;
} zet_metric_group_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricGroupProcAddrTable(
ze_api_version_t version,                       
zet_metric_group_dditable_t* pDdiTable          
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricGroupProcAddrTable_t)(
ze_api_version_t,
zet_metric_group_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGroupCalculateMultipleMetricValuesExp_t)(
zet_metric_group_handle_t,
zet_metric_group_calculation_type_t,
size_t,
const uint8_t*,
uint32_t*,
uint32_t*,
uint32_t*,
zet_typed_value_t*
);

typedef struct _zet_metric_group_exp_dditable_t
{
zet_pfnMetricGroupCalculateMultipleMetricValuesExp_t        pfnCalculateMultipleMetricValuesExp;
} zet_metric_group_exp_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricGroupExpProcAddrTable(
ze_api_version_t version,                       
zet_metric_group_exp_dditable_t* pDdiTable      
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricGroupExpProcAddrTable_t)(
ze_api_version_t,
zet_metric_group_exp_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGet_t)(
zet_metric_group_handle_t,
uint32_t*,
zet_metric_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricGetProperties_t)(
zet_metric_handle_t,
zet_metric_properties_t*
);

typedef struct _zet_metric_dditable_t
{
zet_pfnMetricGet_t                                          pfnGet;
zet_pfnMetricGetProperties_t                                pfnGetProperties;
} zet_metric_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricProcAddrTable(
ze_api_version_t version,                       
zet_metric_dditable_t* pDdiTable                
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricProcAddrTable_t)(
ze_api_version_t,
zet_metric_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricStreamerOpen_t)(
zet_context_handle_t,
zet_device_handle_t,
zet_metric_group_handle_t,
zet_metric_streamer_desc_t*,
ze_event_handle_t,
zet_metric_streamer_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricStreamerClose_t)(
zet_metric_streamer_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricStreamerReadData_t)(
zet_metric_streamer_handle_t,
uint32_t,
size_t*,
uint8_t*
);

typedef struct _zet_metric_streamer_dditable_t
{
zet_pfnMetricStreamerOpen_t                                 pfnOpen;
zet_pfnMetricStreamerClose_t                                pfnClose;
zet_pfnMetricStreamerReadData_t                             pfnReadData;
} zet_metric_streamer_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricStreamerProcAddrTable(
ze_api_version_t version,                       
zet_metric_streamer_dditable_t* pDdiTable       
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricStreamerProcAddrTable_t)(
ze_api_version_t,
zet_metric_streamer_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryPoolCreate_t)(
zet_context_handle_t,
zet_device_handle_t,
zet_metric_group_handle_t,
const zet_metric_query_pool_desc_t*,
zet_metric_query_pool_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryPoolDestroy_t)(
zet_metric_query_pool_handle_t
);

typedef struct _zet_metric_query_pool_dditable_t
{
zet_pfnMetricQueryPoolCreate_t                              pfnCreate;
zet_pfnMetricQueryPoolDestroy_t                             pfnDestroy;
} zet_metric_query_pool_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricQueryPoolProcAddrTable(
ze_api_version_t version,                       
zet_metric_query_pool_dditable_t* pDdiTable     
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricQueryPoolProcAddrTable_t)(
ze_api_version_t,
zet_metric_query_pool_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryCreate_t)(
zet_metric_query_pool_handle_t,
uint32_t,
zet_metric_query_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryDestroy_t)(
zet_metric_query_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryReset_t)(
zet_metric_query_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnMetricQueryGetData_t)(
zet_metric_query_handle_t,
size_t*,
uint8_t*
);

typedef struct _zet_metric_query_dditable_t
{
zet_pfnMetricQueryCreate_t                                  pfnCreate;
zet_pfnMetricQueryDestroy_t                                 pfnDestroy;
zet_pfnMetricQueryReset_t                                   pfnReset;
zet_pfnMetricQueryGetData_t                                 pfnGetData;
} zet_metric_query_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetMetricQueryProcAddrTable(
ze_api_version_t version,                       
zet_metric_query_dditable_t* pDdiTable          
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetMetricQueryProcAddrTable_t)(
ze_api_version_t,
zet_metric_query_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnTracerExpCreate_t)(
zet_context_handle_t,
const zet_tracer_exp_desc_t*,
zet_tracer_exp_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnTracerExpDestroy_t)(
zet_tracer_exp_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnTracerExpSetPrologues_t)(
zet_tracer_exp_handle_t,
zet_core_callbacks_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnTracerExpSetEpilogues_t)(
zet_tracer_exp_handle_t,
zet_core_callbacks_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnTracerExpSetEnabled_t)(
zet_tracer_exp_handle_t,
ze_bool_t
);

typedef struct _zet_tracer_exp_dditable_t
{
zet_pfnTracerExpCreate_t                                    pfnCreate;
zet_pfnTracerExpDestroy_t                                   pfnDestroy;
zet_pfnTracerExpSetPrologues_t                              pfnSetPrologues;
zet_pfnTracerExpSetEpilogues_t                              pfnSetEpilogues;
zet_pfnTracerExpSetEnabled_t                                pfnSetEnabled;
} zet_tracer_exp_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetTracerExpProcAddrTable(
ze_api_version_t version,                       
zet_tracer_exp_dditable_t* pDdiTable            
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetTracerExpProcAddrTable_t)(
ze_api_version_t,
zet_tracer_exp_dditable_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugAttach_t)(
zet_device_handle_t,
const zet_debug_config_t*,
zet_debug_session_handle_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugDetach_t)(
zet_debug_session_handle_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugReadEvent_t)(
zet_debug_session_handle_t,
uint64_t,
zet_debug_event_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugAcknowledgeEvent_t)(
zet_debug_session_handle_t,
const zet_debug_event_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugInterrupt_t)(
zet_debug_session_handle_t,
ze_device_thread_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugResume_t)(
zet_debug_session_handle_t,
ze_device_thread_t
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugReadMemory_t)(
zet_debug_session_handle_t,
ze_device_thread_t,
const zet_debug_memory_space_desc_t*,
size_t,
void*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugWriteMemory_t)(
zet_debug_session_handle_t,
ze_device_thread_t,
const zet_debug_memory_space_desc_t*,
size_t,
const void*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugGetRegisterSetProperties_t)(
zet_device_handle_t,
uint32_t*,
zet_debug_regset_properties_t*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugReadRegisters_t)(
zet_debug_session_handle_t,
ze_device_thread_t,
uint32_t,
uint32_t,
uint32_t,
void*
);

typedef ze_result_t (ZE_APICALL *zet_pfnDebugWriteRegisters_t)(
zet_debug_session_handle_t,
ze_device_thread_t,
uint32_t,
uint32_t,
uint32_t,
void*
);

typedef struct _zet_debug_dditable_t
{
zet_pfnDebugAttach_t                                        pfnAttach;
zet_pfnDebugDetach_t                                        pfnDetach;
zet_pfnDebugReadEvent_t                                     pfnReadEvent;
zet_pfnDebugAcknowledgeEvent_t                              pfnAcknowledgeEvent;
zet_pfnDebugInterrupt_t                                     pfnInterrupt;
zet_pfnDebugResume_t                                        pfnResume;
zet_pfnDebugReadMemory_t                                    pfnReadMemory;
zet_pfnDebugWriteMemory_t                                   pfnWriteMemory;
zet_pfnDebugGetRegisterSetProperties_t                      pfnGetRegisterSetProperties;
zet_pfnDebugReadRegisters_t                                 pfnReadRegisters;
zet_pfnDebugWriteRegisters_t                                pfnWriteRegisters;
} zet_debug_dditable_t;

ZE_DLLEXPORT ze_result_t ZE_APICALL
zetGetDebugProcAddrTable(
ze_api_version_t version,                       
zet_debug_dditable_t* pDdiTable                 
);

typedef ze_result_t (ZE_APICALL *zet_pfnGetDebugProcAddrTable_t)(
ze_api_version_t,
zet_debug_dditable_t*
);

typedef struct _zet_dditable_t
{
zet_device_dditable_t               Device;
zet_context_dditable_t              Context;
zet_command_list_dditable_t         CommandList;
zet_module_dditable_t               Module;
zet_kernel_dditable_t               Kernel;
zet_metric_group_dditable_t         MetricGroup;
zet_metric_group_exp_dditable_t     MetricGroupExp;
zet_metric_dditable_t               Metric;
zet_metric_streamer_dditable_t      MetricStreamer;
zet_metric_query_pool_dditable_t    MetricQueryPool;
zet_metric_query_dditable_t         MetricQuery;
zet_tracer_exp_dditable_t           TracerExp;
zet_debug_dditable_t                Debug;
} zet_dditable_t;

#if defined(__cplusplus)
} 
#endif

#endif 