

#ifndef __OPENCL_CL_D3D10_H
#define __OPENCL_CL_D3D10_H

#if defined(_MSC_VER)
#if _MSC_VER >=1500
#pragma warning( push )
#pragma warning( disable : 4201 )
#endif
#endif
#include <d3d10.h>
#if defined(_MSC_VER)
#if _MSC_VER >=1500
#pragma warning( pop )
#endif
#endif
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif


#define cl_khr_d3d10_sharing 1

typedef cl_uint cl_d3d10_device_source_khr;
typedef cl_uint cl_d3d10_device_set_khr;




#define CL_INVALID_D3D10_DEVICE_KHR                  -1002
#define CL_INVALID_D3D10_RESOURCE_KHR                -1003
#define CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR       -1004
#define CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR           -1005


#define CL_D3D10_DEVICE_KHR                          0x4010
#define CL_D3D10_DXGI_ADAPTER_KHR                    0x4011


#define CL_PREFERRED_DEVICES_FOR_D3D10_KHR           0x4012
#define CL_ALL_DEVICES_FOR_D3D10_KHR                 0x4013


#define CL_CONTEXT_D3D10_DEVICE_KHR                  0x4014
#define CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR 0x402C


#define CL_MEM_D3D10_RESOURCE_KHR                    0x4015


#define CL_IMAGE_D3D10_SUBRESOURCE_KHR               0x4016


#define CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR         0x4017
#define CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR         0x4018



typedef cl_int (CL_API_CALL *clGetDeviceIDsFromD3D10KHR_fn)(
cl_platform_id             platform,
cl_d3d10_device_source_khr d3d_device_source,
void *                     d3d_object,
cl_d3d10_device_set_khr    d3d_device_set,
cl_uint                    num_entries,
cl_device_id *             devices,
cl_uint *                  num_devices) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10BufferKHR_fn)(
cl_context     context,
cl_mem_flags   flags,
ID3D10Buffer * resource,
cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10Texture2DKHR_fn)(
cl_context        context,
cl_mem_flags      flags,
ID3D10Texture2D * resource,
UINT              subresource,
cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10Texture3DKHR_fn)(
cl_context        context,
cl_mem_flags      flags,
ID3D10Texture3D * resource,
UINT              subresource,
cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *clEnqueueAcquireD3D10ObjectsKHR_fn)(
cl_command_queue command_queue,
cl_uint          num_objects,
const cl_mem *   mem_objects,
cl_uint          num_events_in_wait_list,
const cl_event * event_wait_list,
cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *clEnqueueReleaseD3D10ObjectsKHR_fn)(
cl_command_queue command_queue,
cl_uint          num_objects,
const cl_mem *   mem_objects,
cl_uint          num_events_in_wait_list,
const cl_event * event_wait_list,
cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
}
#endif

#endif  

