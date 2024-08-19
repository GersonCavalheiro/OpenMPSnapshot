

#ifndef _ZEL_TRACING_DDI_H
#define _ZEL_TRACING_DDI_H
#if defined(__cplusplus)
#pragma once
#endif
#include "layers/zel_tracing_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef ze_result_t (ZE_APICALL *zel_pfnTracerCreate_t)(
const zel_tracer_desc_t*,
zel_tracer_handle_t*
);

typedef ze_result_t (ZE_APICALL *zel_pfnTracerDestroy_t)(
zel_tracer_handle_t
);

typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetPrologues_t)(
zel_tracer_handle_t,
zel_core_callbacks_t*
);

typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetEpilogues_t)(
zel_tracer_handle_t,
zel_core_callbacks_t*
);

typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetEnabled_t)(
zel_tracer_handle_t,
ze_bool_t
);


typedef struct _zel_tracer_dditable_t
{
zel_pfnTracerCreate_t                                    pfnCreate;
zel_pfnTracerDestroy_t                                   pfnDestroy;
zel_pfnTracerSetPrologues_t                              pfnSetPrologues;
zel_pfnTracerSetEpilogues_t                              pfnSetEpilogues;
zel_pfnTracerSetEnabled_t                                pfnSetEnabled;
} zel_tracer_dditable_t;


ZE_DLLEXPORT ze_result_t ZE_APICALL
zelGetTracerApiProcAddrTable(
ze_api_version_t version,                       
zel_tracer_dditable_t* pDdiTable            
);

typedef ze_result_t (ZE_APICALL *zel_pfnGetTracerApiProcAddrTable_t)(
ze_api_version_t,
zel_tracer_dditable_t*
);

typedef struct _zel_tracing_dditable_t
{
zel_tracer_dditable_t         Tracer;
} zel_tracing_dditable_t;

#if defined(__cplusplus)
} 
#endif

#endif 
