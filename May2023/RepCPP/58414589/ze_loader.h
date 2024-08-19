

#ifndef _ZE_LOADER_H
#define _ZE_LOADER_H
#if defined(__cplusplus)
#pragma once
#endif

#include "../ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct _zel_version {
int major;
int minor;
int patch; 
} zel_version_t; 

#define ZEL_COMPONENT_STRING_SIZE 64 

typedef struct zel_component_version {
char component_name[ZEL_COMPONENT_STRING_SIZE];
ze_api_version_t spec_version;
zel_version_t component_lib_version;
} zel_component_version_t; 


ZE_APIEXPORT ze_result_t ZE_APICALL
zelLoaderGetVersions(
size_t *num_elems,                     
zel_component_version_t *versions);    

typedef enum _zel_handle_type_t {
ZEL_HANDLE_DRIVER,
ZEL_HANDLE_DEVICE,
ZEL_HANDLE_CONTEXT,
ZEL_HANDLE_COMMAND_QUEUE,
ZEL_HANDLE_COMMAND_LIST,
ZEL_HANDLE_FENCE,
ZEL_HANDLE_EVENT_POOL,
ZEL_HANDLE_EVENT,
ZEL_HANDLE_IMAGE,
ZEL_HANDLE_MODULE,
ZEL_HANDLE_MODULE_BUILD_LOG,
ZEL_HANDLE_KERNEL,
ZEL_HANDLE_SAMPLER,
ZEL_HANDLE_PHYSICAL_MEM
} zel_handle_type_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zelLoaderTranslateHandle(
zel_handle_type_t handleType,   
void *handleIn,                  
void **handleOut);                


#if defined(__cplusplus)
} 
#endif
#endif 