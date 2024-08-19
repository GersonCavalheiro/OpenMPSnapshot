
#ifndef _ZEL_TRACING_API_H
#define _ZEL_TRACING_API_H
#if defined(__cplusplus)
#pragma once
#endif

#include "../ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(__GNUC__)
#pragma region zel_tracing
#endif

typedef struct _zel_tracer_handle_t *zel_tracer_handle_t;

#ifndef ZEL_API_TRACING_NAME
#define ZEL_API_TRACING_NAME  "ZEL_api_tracing"
#endif 

typedef enum _zel_api_tracing_version_t
{
ZEL_API_TRACING_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  
ZEL_API_TRACING_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  
ZEL_API_TRACING_VERSION_FORCE_UINT32 = 0x7fffffff

} zel_api_tracing_version_t;

typedef ze_callbacks_t zel_core_callbacks_t;

typedef enum _zel_structure_type_t
{
ZEL_STRUCTURE_TYPE_TRACER_DESC = 0x1  ,
ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC = 0x1  ,
ZEL_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zel_structure_type_t;

typedef struct _zel_tracer_desc_t
{
zel_structure_type_t stype;                     
const void* pNext;                              
void* pUserData;                                

} zel_tracer_desc_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerCreate(
const zel_tracer_desc_t* desc,              
zel_tracer_handle_t* phTracer               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerDestroy(
zel_tracer_handle_t hTracer                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetPrologues(
zel_tracer_handle_t hTracer,                
zel_core_callbacks_t* pCoreCbs                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetEpilogues(
zel_tracer_handle_t hTracer,                
zel_core_callbacks_t* pCoreCbs                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetEnabled(
zel_tracer_handle_t hTracer,                
ze_bool_t enable                                
);

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} 
#endif

#endif 