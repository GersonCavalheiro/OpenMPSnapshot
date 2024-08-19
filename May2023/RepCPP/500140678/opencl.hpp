



#ifndef CL_HPP_
#define CL_HPP_


#if !defined(CL_HPP_USE_DX_INTEROP) && defined(USE_DX_INTEROP)
# pragma message("opencl.hpp: USE_DX_INTEROP is deprecated. Define CL_HPP_USE_DX_INTEROP instead")
# define CL_HPP_USE_DX_INTEROP
#endif
#if !defined(CL_HPP_USE_CL_DEVICE_FISSION) && defined(USE_CL_DEVICE_FISSION)
# pragma message("opencl.hpp: USE_CL_DEVICE_FISSION is deprecated. Define CL_HPP_USE_CL_DEVICE_FISSION instead")
# define CL_HPP_USE_CL_DEVICE_FISSION
#endif
#if !defined(CL_HPP_ENABLE_EXCEPTIONS) && defined(__CL_ENABLE_EXCEPTIONS)
# pragma message("opencl.hpp: __CL_ENABLE_EXCEPTIONS is deprecated. Define CL_HPP_ENABLE_EXCEPTIONS instead")
# define CL_HPP_ENABLE_EXCEPTIONS
#endif
#if !defined(CL_HPP_NO_STD_VECTOR) && defined(__NO_STD_VECTOR)
# pragma message("opencl.hpp: __NO_STD_VECTOR is deprecated. Define CL_HPP_NO_STD_VECTOR instead")
# define CL_HPP_NO_STD_VECTOR
#endif
#if !defined(CL_HPP_NO_STD_STRING) && defined(__NO_STD_STRING)
# pragma message("opencl.hpp: __NO_STD_STRING is deprecated. Define CL_HPP_NO_STD_STRING instead")
# define CL_HPP_NO_STD_STRING
#endif
#if defined(VECTOR_CLASS)
# pragma message("opencl.hpp: VECTOR_CLASS is deprecated. Alias cl::vector instead")
#endif
#if defined(STRING_CLASS)
# pragma message("opencl.hpp: STRING_CLASS is deprecated. Alias cl::string instead.")
#endif
#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS) && defined(__CL_USER_OVERRIDE_ERROR_STRINGS)
# pragma message("opencl.hpp: __CL_USER_OVERRIDE_ERROR_STRINGS is deprecated. Define CL_HPP_USER_OVERRIDE_ERROR_STRINGS instead")
# define CL_HPP_USER_OVERRIDE_ERROR_STRINGS
#endif


#if defined(__USE_DEV_VECTOR)
# pragma message("opencl.hpp: __USE_DEV_VECTOR is no longer supported. Expect compilation errors")
#endif
#if defined(__USE_DEV_STRING)
# pragma message("opencl.hpp: __USE_DEV_STRING is no longer supported. Expect compilation errors")
#endif


#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 300 (OpenCL 3.0)")
# define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#if CL_HPP_TARGET_OPENCL_VERSION != 100 && \
CL_HPP_TARGET_OPENCL_VERSION != 110 && \
CL_HPP_TARGET_OPENCL_VERSION != 120 && \
CL_HPP_TARGET_OPENCL_VERSION != 200 && \
CL_HPP_TARGET_OPENCL_VERSION != 210 && \
CL_HPP_TARGET_OPENCL_VERSION != 220 && \
CL_HPP_TARGET_OPENCL_VERSION != 300
# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 300 (OpenCL 3.0).")
# undef CL_HPP_TARGET_OPENCL_VERSION
# define CL_HPP_TARGET_OPENCL_VERSION 300
#endif


#if defined(CL_TARGET_OPENCL_VERSION)

#if CL_TARGET_OPENCL_VERSION < CL_HPP_TARGET_OPENCL_VERSION
# pragma message("CL_TARGET_OPENCL_VERSION is already defined as is lower than CL_HPP_TARGET_OPENCL_VERSION")
#endif
#else
# define CL_TARGET_OPENCL_VERSION CL_HPP_TARGET_OPENCL_VERSION
#endif

#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
# define CL_HPP_MINIMUM_OPENCL_VERSION 200
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 110 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 120 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 200 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 210 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 220 && \
CL_HPP_MINIMUM_OPENCL_VERSION != 300
# pragma message("opencl.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 100")
# undef CL_HPP_MINIMUM_OPENCL_VERSION
# define CL_HPP_MINIMUM_OPENCL_VERSION 100
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
#endif

#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 210 && !defined(CL_USE_DEPRECATED_OPENCL_2_1_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_1_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 220 && !defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_2_APIS
#endif

#ifdef _WIN32

#include <malloc.h>

#if defined(CL_HPP_USE_DX_INTEROP)
#include <CL/cl_d3d10.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#endif 

#if defined(_MSC_VER)
#include <intrin.h>
#endif 


#if (!defined(_MSC_VER) && __cplusplus < 201103L) || (defined(_MSC_VER) && _MSC_VER < 1700)
#error Visual studio 2013 or another C++11-supporting compiler required
#endif

#if defined(CL_HPP_USE_CL_DEVICE_FISSION) || defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)
#include <CL/cl_ext.h>
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif 

#if (__cplusplus >= 201103L || _MSVC_LANG >= 201103L )
#define CL_HPP_NOEXCEPT_ noexcept
#else
#define CL_HPP_NOEXCEPT_
#endif

#if __cplusplus >= 201703L
# define CL_HPP_DEFINE_STATIC_MEMBER_ inline
#elif defined(_MSC_VER)
# define CL_HPP_DEFINE_STATIC_MEMBER_ __declspec(selectany)
#elif defined(__MINGW32__)
# define CL_HPP_DEFINE_STATIC_MEMBER_ __attribute__((selectany))
#else
# define CL_HPP_DEFINE_STATIC_MEMBER_ __attribute__((weak))
#endif 

#if !defined(CL_API_PREFIX__VERSION_1_1_DEPRECATED)
#define CL_API_PREFIX__VERSION_1_1_DEPRECATED
#endif 
#if !defined(CL_API_SUFFIX__VERSION_1_1_DEPRECATED)
#define CL_API_SUFFIX__VERSION_1_1_DEPRECATED
#endif 

#if !defined(CL_API_PREFIX__VERSION_1_2_DEPRECATED)
#define CL_API_PREFIX__VERSION_1_2_DEPRECATED
#endif 
#if !defined(CL_API_SUFFIX__VERSION_1_2_DEPRECATED)
#define CL_API_SUFFIX__VERSION_1_2_DEPRECATED
#endif 

#if !defined(CL_API_PREFIX__VERSION_2_2_DEPRECATED)
#define CL_API_PREFIX__VERSION_2_2_DEPRECATED
#endif 
#if !defined(CL_API_SUFFIX__VERSION_2_2_DEPRECATED)
#define CL_API_SUFFIX__VERSION_2_2_DEPRECATED
#endif 

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif 

#include <utility>
#include <limits>
#include <iterator>
#include <mutex>
#include <cstring>
#include <functional>


#if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)
namespace cl {
using size_type = ::size_t;
} 
#else 
namespace cl {
using size_type = size_t;
} 
#endif 


#if defined(CL_HPP_ENABLE_EXCEPTIONS)
#include <exception>
#endif 

#if !defined(CL_HPP_NO_STD_VECTOR)
#include <vector>
namespace cl {
template < class T, class Alloc = std::allocator<T> >
using vector = std::vector<T, Alloc>;
} 
#endif 

#if !defined(CL_HPP_NO_STD_STRING)
#include <string>
namespace cl {
using string = std::string;
} 
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if !defined(CL_HPP_NO_STD_UNIQUE_PTR)
#include <memory>
namespace cl {
template<class T, class D>
using pointer = std::unique_ptr<T, D>;
} 
#endif 
#endif 
#if !defined(CL_HPP_NO_STD_ARRAY)
#include <array>
namespace cl {
template < class T, size_type N >
using array = std::array<T, N>;
} 
#endif 

#if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)
namespace cl {
namespace compatibility {

template <int N>
class size_t
{
private:
size_type data_[N];

public:
size_t()
{
for (int i = 0; i < N; ++i) {
data_[i] = 0;
}
}

size_t(const array<size_type, N> &rhs)
{
for (int i = 0; i < N; ++i) {
data_[i] = rhs[i];
}
}

size_type& operator[](int index)
{
return data_[index];
}

const size_type& operator[](int index) const
{
return data_[index];
}

operator size_type* ()             { return data_; }

operator const size_type* () const { return data_; }

operator array<size_type, N>() const
{
array<size_type, N> ret;

for (int i = 0; i < N; ++i) {
ret[i] = data_[i];
}
return ret;
}
};
} 

template<int N>
using size_t = compatibility::size_t<N>;
} 
#endif 

namespace cl {
namespace detail {
using size_t_array = array<size_type, 3>;
} 
} 



namespace cl {
class Memory;

#define CL_HPP_INIT_CL_EXT_FCN_PTR_(name) \
if (!pfn_##name) {    \
pfn_##name = (PFN_##name) \
clGetExtensionFunctionAddress(#name); \
if (!pfn_##name) {    \
} \
}

#define CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, name) \
if (!pfn_##name) {    \
pfn_##name = (PFN_##name) \
clGetExtensionFunctionAddressForPlatform(platform, #name); \
if (!pfn_##name) {    \
} \
}

class Program;
class Device;
class Context;
class CommandQueue;
class DeviceCommandQueue;
class Memory;
class Buffer;
class Pipe;

#if defined(CL_HPP_ENABLE_EXCEPTIONS)

class Error : public std::exception
{
private:
cl_int err_;
const char * errStr_;
public:

Error(cl_int err, const char * errStr = NULL) : err_(err), errStr_(errStr)
{}

~Error() throw() {}


virtual const char * what() const throw ()
{
if (errStr_ == NULL) {
return "empty";
}
else {
return errStr_;
}
}


cl_int err(void) const { return err_; }
};
#define CL_HPP_ERR_STR_(x) #x
#else
#define CL_HPP_ERR_STR_(x) NULL
#endif 


namespace detail
{
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
static inline cl_int errHandler (
cl_int err,
const char * errStr = NULL)
{
if (err != CL_SUCCESS) {
throw Error(err, errStr);
}
return err;
}
#else
static inline cl_int errHandler (cl_int err, const char * errStr = NULL)
{
(void) errStr; 
return err;
}
#endif 
}



#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS)
#define __GET_DEVICE_INFO_ERR               CL_HPP_ERR_STR_(clGetDeviceInfo)
#define __GET_PLATFORM_INFO_ERR             CL_HPP_ERR_STR_(clGetPlatformInfo)
#define __GET_DEVICE_IDS_ERR                CL_HPP_ERR_STR_(clGetDeviceIDs)
#define __GET_PLATFORM_IDS_ERR              CL_HPP_ERR_STR_(clGetPlatformIDs)
#define __GET_CONTEXT_INFO_ERR              CL_HPP_ERR_STR_(clGetContextInfo)
#define __GET_EVENT_INFO_ERR                CL_HPP_ERR_STR_(clGetEventInfo)
#define __GET_EVENT_PROFILE_INFO_ERR        CL_HPP_ERR_STR_(clGetEventProfileInfo)
#define __GET_MEM_OBJECT_INFO_ERR           CL_HPP_ERR_STR_(clGetMemObjectInfo)
#define __GET_IMAGE_INFO_ERR                CL_HPP_ERR_STR_(clGetImageInfo)
#define __GET_SAMPLER_INFO_ERR              CL_HPP_ERR_STR_(clGetSamplerInfo)
#define __GET_KERNEL_INFO_ERR               CL_HPP_ERR_STR_(clGetKernelInfo)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __GET_KERNEL_ARG_INFO_ERR           CL_HPP_ERR_STR_(clGetKernelArgInfo)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
#define __GET_KERNEL_SUB_GROUP_INFO_ERR     CL_HPP_ERR_STR_(clGetKernelSubGroupInfo)
#endif 
#define __GET_KERNEL_WORK_GROUP_INFO_ERR    CL_HPP_ERR_STR_(clGetKernelWorkGroupInfo)
#define __GET_PROGRAM_INFO_ERR              CL_HPP_ERR_STR_(clGetProgramInfo)
#define __GET_PROGRAM_BUILD_INFO_ERR        CL_HPP_ERR_STR_(clGetProgramBuildInfo)
#define __GET_COMMAND_QUEUE_INFO_ERR        CL_HPP_ERR_STR_(clGetCommandQueueInfo)

#define __CREATE_CONTEXT_ERR                CL_HPP_ERR_STR_(clCreateContext)
#define __CREATE_CONTEXT_FROM_TYPE_ERR      CL_HPP_ERR_STR_(clCreateContextFromType)
#define __GET_SUPPORTED_IMAGE_FORMATS_ERR   CL_HPP_ERR_STR_(clGetSupportedImageFormats)

#define __CREATE_BUFFER_ERR                 CL_HPP_ERR_STR_(clCreateBuffer)
#define __COPY_ERR                          CL_HPP_ERR_STR_(cl::copy)
#define __CREATE_SUBBUFFER_ERR              CL_HPP_ERR_STR_(clCreateSubBuffer)
#define __CREATE_GL_BUFFER_ERR              CL_HPP_ERR_STR_(clCreateFromGLBuffer)
#define __CREATE_GL_RENDER_BUFFER_ERR       CL_HPP_ERR_STR_(clCreateFromGLBuffer)
#define __GET_GL_OBJECT_INFO_ERR            CL_HPP_ERR_STR_(clGetGLObjectInfo)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_IMAGE_ERR                  CL_HPP_ERR_STR_(clCreateImage)
#define __CREATE_GL_TEXTURE_ERR             CL_HPP_ERR_STR_(clCreateFromGLTexture)
#define __IMAGE_DIMENSION_ERR               CL_HPP_ERR_STR_(Incorrect image dimensions)
#endif 
#define __SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR CL_HPP_ERR_STR_(clSetMemObjectDestructorCallback)

#define __CREATE_USER_EVENT_ERR             CL_HPP_ERR_STR_(clCreateUserEvent)
#define __SET_USER_EVENT_STATUS_ERR         CL_HPP_ERR_STR_(clSetUserEventStatus)
#define __SET_EVENT_CALLBACK_ERR            CL_HPP_ERR_STR_(clSetEventCallback)
#define __WAIT_FOR_EVENTS_ERR               CL_HPP_ERR_STR_(clWaitForEvents)

#define __CREATE_KERNEL_ERR                 CL_HPP_ERR_STR_(clCreateKernel)
#define __SET_KERNEL_ARGS_ERR               CL_HPP_ERR_STR_(clSetKernelArg)
#define __CREATE_PROGRAM_WITH_SOURCE_ERR    CL_HPP_ERR_STR_(clCreateProgramWithSource)
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
#define __CREATE_PROGRAM_WITH_IL_ERR        CL_HPP_ERR_STR_(clCreateProgramWithIL)
#endif 
#define __CREATE_PROGRAM_WITH_BINARY_ERR    CL_HPP_ERR_STR_(clCreateProgramWithBinary)
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __CREATE_PROGRAM_WITH_IL_ERR        CL_HPP_ERR_STR_(clCreateProgramWithIL)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR    CL_HPP_ERR_STR_(clCreateProgramWithBuiltInKernels)
#endif 
#define __BUILD_PROGRAM_ERR                 CL_HPP_ERR_STR_(clBuildProgram)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __COMPILE_PROGRAM_ERR               CL_HPP_ERR_STR_(clCompileProgram)
#define __LINK_PROGRAM_ERR                  CL_HPP_ERR_STR_(clLinkProgram)
#endif 
#define __CREATE_KERNELS_IN_PROGRAM_ERR     CL_HPP_ERR_STR_(clCreateKernelsInProgram)

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
#define __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR          CL_HPP_ERR_STR_(clCreateCommandQueueWithProperties)
#define __CREATE_SAMPLER_WITH_PROPERTIES_ERR                CL_HPP_ERR_STR_(clCreateSamplerWithProperties)
#endif 
#define __SET_COMMAND_QUEUE_PROPERTY_ERR    CL_HPP_ERR_STR_(clSetCommandQueueProperty)
#define __ENQUEUE_READ_BUFFER_ERR           CL_HPP_ERR_STR_(clEnqueueReadBuffer)
#define __ENQUEUE_READ_BUFFER_RECT_ERR      CL_HPP_ERR_STR_(clEnqueueReadBufferRect)
#define __ENQUEUE_WRITE_BUFFER_ERR          CL_HPP_ERR_STR_(clEnqueueWriteBuffer)
#define __ENQUEUE_WRITE_BUFFER_RECT_ERR     CL_HPP_ERR_STR_(clEnqueueWriteBufferRect)
#define __ENQEUE_COPY_BUFFER_ERR            CL_HPP_ERR_STR_(clEnqueueCopyBuffer)
#define __ENQEUE_COPY_BUFFER_RECT_ERR       CL_HPP_ERR_STR_(clEnqueueCopyBufferRect)
#define __ENQUEUE_FILL_BUFFER_ERR           CL_HPP_ERR_STR_(clEnqueueFillBuffer)
#define __ENQUEUE_READ_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueReadImage)
#define __ENQUEUE_WRITE_IMAGE_ERR           CL_HPP_ERR_STR_(clEnqueueWriteImage)
#define __ENQUEUE_COPY_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueCopyImage)
#define __ENQUEUE_FILL_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueFillImage)
#define __ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR  CL_HPP_ERR_STR_(clEnqueueCopyImageToBuffer)
#define __ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR  CL_HPP_ERR_STR_(clEnqueueCopyBufferToImage)
#define __ENQUEUE_MAP_BUFFER_ERR            CL_HPP_ERR_STR_(clEnqueueMapBuffer)
#define __ENQUEUE_MAP_IMAGE_ERR             CL_HPP_ERR_STR_(clEnqueueMapImage)
#define __ENQUEUE_UNMAP_MEM_OBJECT_ERR      CL_HPP_ERR_STR_(clEnqueueUnMapMemObject)
#define __ENQUEUE_NDRANGE_KERNEL_ERR        CL_HPP_ERR_STR_(clEnqueueNDRangeKernel)
#define __ENQUEUE_NATIVE_KERNEL             CL_HPP_ERR_STR_(clEnqueueNativeKernel)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __ENQUEUE_MIGRATE_MEM_OBJECTS_ERR   CL_HPP_ERR_STR_(clEnqueueMigrateMemObjects)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __ENQUEUE_MIGRATE_SVM_ERR   CL_HPP_ERR_STR_(clEnqueueSVMMigrateMem)
#define __SET_DEFAULT_DEVICE_COMMAND_QUEUE_ERR   CL_HPP_ERR_STR_(clSetDefaultDeviceCommandQueue)
#endif 


#define __ENQUEUE_ACQUIRE_GL_ERR            CL_HPP_ERR_STR_(clEnqueueAcquireGLObjects)
#define __ENQUEUE_RELEASE_GL_ERR            CL_HPP_ERR_STR_(clEnqueueReleaseGLObjects)

#define __CREATE_PIPE_ERR             CL_HPP_ERR_STR_(clCreatePipe)
#define __GET_PIPE_INFO_ERR           CL_HPP_ERR_STR_(clGetPipeInfo)


#define __RETAIN_ERR                        CL_HPP_ERR_STR_(Retain Object)
#define __RELEASE_ERR                       CL_HPP_ERR_STR_(Release Object)
#define __FLUSH_ERR                         CL_HPP_ERR_STR_(clFlush)
#define __FINISH_ERR                        CL_HPP_ERR_STR_(clFinish)
#define __VECTOR_CAPACITY_ERR               CL_HPP_ERR_STR_(Vector capacity error)

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __GET_HOST_TIMER_ERR           CL_HPP_ERR_STR_(clGetHostTimer)
#define __GET_DEVICE_AND_HOST_TIMER_ERR           CL_HPP_ERR_STR_(clGetDeviceAndHostTimer)
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 220
#define __SET_PROGRAM_RELEASE_CALLBACK_ERR          CL_HPP_ERR_STR_(clSetProgramReleaseCallback)
#define __SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR   CL_HPP_ERR_STR_(clSetProgramSpecializationConstant)
#endif



#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_SUB_DEVICES_ERR            CL_HPP_ERR_STR_(clCreateSubDevices)
#else
#define __CREATE_SUB_DEVICES_ERR            CL_HPP_ERR_STR_(clCreateSubDevicesEXT)
#endif 


#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
#define __ENQUEUE_MARKER_ERR                CL_HPP_ERR_STR_(clEnqueueMarker)
#define __ENQUEUE_WAIT_FOR_EVENTS_ERR       CL_HPP_ERR_STR_(clEnqueueWaitForEvents)
#define __ENQUEUE_BARRIER_ERR               CL_HPP_ERR_STR_(clEnqueueBarrier)
#define __UNLOAD_COMPILER_ERR               CL_HPP_ERR_STR_(clUnloadCompiler)
#define __CREATE_GL_TEXTURE_2D_ERR          CL_HPP_ERR_STR_(clCreateFromGLTexture2D)
#define __CREATE_GL_TEXTURE_3D_ERR          CL_HPP_ERR_STR_(clCreateFromGLTexture3D)
#define __CREATE_IMAGE2D_ERR                CL_HPP_ERR_STR_(clCreateImage2D)
#define __CREATE_IMAGE3D_ERR                CL_HPP_ERR_STR_(clCreateImage3D)
#endif 


#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#define __CREATE_COMMAND_QUEUE_ERR          CL_HPP_ERR_STR_(clCreateCommandQueue)
#define __ENQUEUE_TASK_ERR                  CL_HPP_ERR_STR_(clEnqueueTask)
#define __CREATE_SAMPLER_ERR                CL_HPP_ERR_STR_(clCreateSampler)
#endif 


#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __ENQUEUE_MARKER_WAIT_LIST_ERR                CL_HPP_ERR_STR_(clEnqueueMarkerWithWaitList)
#define __ENQUEUE_BARRIER_WAIT_LIST_ERR               CL_HPP_ERR_STR_(clEnqueueBarrierWithWaitList)
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __CLONE_KERNEL_ERR     CL_HPP_ERR_STR_(clCloneKernel)
#endif 

#endif 


namespace detail {

template<typename Functor, typename T>
inline cl_int getInfoHelper(Functor f, cl_uint name, T* param, long)
{
return f(name, sizeof(T), param, NULL);
}

template <typename Func>
inline cl_int getInfoHelper(Func f, cl_uint name, vector<vector<unsigned char>>* param, int)
{
if (name != CL_PROGRAM_BINARIES) {
return CL_INVALID_VALUE;
}
if (param) {
size_type numBinaries = param->size();
vector<unsigned char*> binariesPointers(numBinaries);

for (size_type i = 0; i < numBinaries; ++i)
{
binariesPointers[i] = (*param)[i].data();
}

cl_int err = f(name, numBinaries * sizeof(unsigned char*), binariesPointers.data(), NULL);

if (err != CL_SUCCESS) {
return err;
}
}


return CL_SUCCESS;
}

template <typename Func, typename T>
inline cl_int getInfoHelper(Func f, cl_uint name, vector<T>* param, long)
{
size_type required;
cl_int err = f(name, 0, NULL, &required);
if (err != CL_SUCCESS) {
return err;
}
const size_type elements = required / sizeof(T);

vector<T> localData(elements);
err = f(name, required, localData.data(), NULL);
if (err != CL_SUCCESS) {
return err;
}
if (param) {
*param = std::move(localData);
}

return CL_SUCCESS;
}


template <typename Func, typename T>
inline cl_int getInfoHelper(
Func f, cl_uint name, vector<T>* param, int, typename T::cl_type = 0)
{
size_type required;
cl_int err = f(name, 0, NULL, &required);
if (err != CL_SUCCESS) {
return err;
}

const size_type elements = required / sizeof(typename T::cl_type);

vector<typename T::cl_type> value(elements);
err = f(name, required, value.data(), NULL);
if (err != CL_SUCCESS) {
return err;
}

if (param) {
param->resize(elements);

for (size_type i = 0; i < elements; i++) {
(*param)[i] = T(value[i], true);
}
}
return CL_SUCCESS;
}

template <typename Func>
inline cl_int getInfoHelper(Func f, cl_uint name, string* param, long)
{
size_type required;
cl_int err = f(name, 0, NULL, &required);
if (err != CL_SUCCESS) {
return err;
}

if (required > 0) {
vector<char> value(required);
err = f(name, required, value.data(), NULL);
if (err != CL_SUCCESS) {
return err;
}
if (param) {
param->assign(begin(value), prev(end(value)));
}
}
else if (param) {
param->assign("");
}
return CL_SUCCESS;
}

template <typename Func, size_type N>
inline cl_int getInfoHelper(Func f, cl_uint name, array<size_type, N>* param, long)
{
size_type required;
cl_int err = f(name, 0, NULL, &required);
if (err != CL_SUCCESS) {
return err;
}

size_type elements = required / sizeof(size_type);
vector<size_type> value(elements, 0);

err = f(name, required, value.data(), NULL);
if (err != CL_SUCCESS) {
return err;
}

if (elements > N) {
elements = N;
}
for (size_type i = 0; i < elements; ++i) {
(*param)[i] = value[i];
}

return CL_SUCCESS;
}

template<typename T> struct ReferenceHandler;


template<typename Func, typename T>
inline cl_int getInfoHelper(Func f, cl_uint name, T* param, int, typename T::cl_type = 0)
{
typename T::cl_type value;
cl_int err = f(name, sizeof(value), &value, NULL);
if (err != CL_SUCCESS) {
return err;
}
*param = value;
if (value != NULL)
{
err = param->retain();
if (err != CL_SUCCESS) {
return err;
}
}
return CL_SUCCESS;
}

#define CL_HPP_PARAM_NAME_INFO_1_0_(F) \
F(cl_platform_info, CL_PLATFORM_PROFILE, string) \
F(cl_platform_info, CL_PLATFORM_VERSION, string) \
F(cl_platform_info, CL_PLATFORM_NAME, string) \
F(cl_platform_info, CL_PLATFORM_VENDOR, string) \
F(cl_platform_info, CL_PLATFORM_EXTENSIONS, string) \
\
F(cl_device_info, CL_DEVICE_TYPE, cl_device_type) \
F(cl_device_info, CL_DEVICE_VENDOR_ID, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_WORK_GROUP_SIZE, size_type) \
F(cl_device_info, CL_DEVICE_MAX_WORK_ITEM_SIZES, cl::vector<size_type>) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint) \
F(cl_device_info, CL_DEVICE_ADDRESS_BITS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong) \
F(cl_device_info, CL_DEVICE_IMAGE2D_MAX_WIDTH, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE2D_MAX_HEIGHT, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_WIDTH, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_HEIGHT, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_DEPTH, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_MAX_PARAMETER_SIZE, size_type) \
F(cl_device_info, CL_DEVICE_MAX_SAMPLERS, cl_uint) \
F(cl_device_info, CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint) \
F(cl_device_info, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, cl_uint) \
F(cl_device_info, CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config) \
F(cl_device_info, CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config) \
F(cl_device_info, CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config) \
F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type) \
F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint)\
F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong) \
F(cl_device_info, CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong) \
F(cl_device_info, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong) \
F(cl_device_info, CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint) \
F(cl_device_info, CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type) \
F(cl_device_info, CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong) \
F(cl_device_info, CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_type) \
F(cl_device_info, CL_DEVICE_ENDIAN_LITTLE, cl_bool) \
F(cl_device_info, CL_DEVICE_AVAILABLE, cl_bool) \
F(cl_device_info, CL_DEVICE_COMPILER_AVAILABLE, cl_bool) \
F(cl_device_info, CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities) \
F(cl_device_info, CL_DEVICE_PLATFORM, cl_platform_id) \
F(cl_device_info, CL_DEVICE_NAME, string) \
F(cl_device_info, CL_DEVICE_VENDOR, string) \
F(cl_device_info, CL_DRIVER_VERSION, string) \
F(cl_device_info, CL_DEVICE_PROFILE, string) \
F(cl_device_info, CL_DEVICE_VERSION, string) \
F(cl_device_info, CL_DEVICE_EXTENSIONS, string) \
\
F(cl_context_info, CL_CONTEXT_REFERENCE_COUNT, cl_uint) \
F(cl_context_info, CL_CONTEXT_DEVICES, cl::vector<Device>) \
F(cl_context_info, CL_CONTEXT_PROPERTIES, cl::vector<cl_context_properties>) \
\
F(cl_event_info, CL_EVENT_COMMAND_QUEUE, cl::CommandQueue) \
F(cl_event_info, CL_EVENT_COMMAND_TYPE, cl_command_type) \
F(cl_event_info, CL_EVENT_REFERENCE_COUNT, cl_uint) \
F(cl_event_info, CL_EVENT_COMMAND_EXECUTION_STATUS, cl_int) \
\
F(cl_profiling_info, CL_PROFILING_COMMAND_QUEUED, cl_ulong) \
F(cl_profiling_info, CL_PROFILING_COMMAND_SUBMIT, cl_ulong) \
F(cl_profiling_info, CL_PROFILING_COMMAND_START, cl_ulong) \
F(cl_profiling_info, CL_PROFILING_COMMAND_END, cl_ulong) \
\
F(cl_mem_info, CL_MEM_TYPE, cl_mem_object_type) \
F(cl_mem_info, CL_MEM_FLAGS, cl_mem_flags) \
F(cl_mem_info, CL_MEM_SIZE, size_type) \
F(cl_mem_info, CL_MEM_HOST_PTR, void*) \
F(cl_mem_info, CL_MEM_MAP_COUNT, cl_uint) \
F(cl_mem_info, CL_MEM_REFERENCE_COUNT, cl_uint) \
F(cl_mem_info, CL_MEM_CONTEXT, cl::Context) \
\
F(cl_image_info, CL_IMAGE_FORMAT, cl_image_format) \
F(cl_image_info, CL_IMAGE_ELEMENT_SIZE, size_type) \
F(cl_image_info, CL_IMAGE_ROW_PITCH, size_type) \
F(cl_image_info, CL_IMAGE_SLICE_PITCH, size_type) \
F(cl_image_info, CL_IMAGE_WIDTH, size_type) \
F(cl_image_info, CL_IMAGE_HEIGHT, size_type) \
F(cl_image_info, CL_IMAGE_DEPTH, size_type) \
\
F(cl_sampler_info, CL_SAMPLER_REFERENCE_COUNT, cl_uint) \
F(cl_sampler_info, CL_SAMPLER_CONTEXT, cl::Context) \
F(cl_sampler_info, CL_SAMPLER_NORMALIZED_COORDS, cl_bool) \
F(cl_sampler_info, CL_SAMPLER_ADDRESSING_MODE, cl_addressing_mode) \
F(cl_sampler_info, CL_SAMPLER_FILTER_MODE, cl_filter_mode) \
\
F(cl_program_info, CL_PROGRAM_REFERENCE_COUNT, cl_uint) \
F(cl_program_info, CL_PROGRAM_CONTEXT, cl::Context) \
F(cl_program_info, CL_PROGRAM_NUM_DEVICES, cl_uint) \
F(cl_program_info, CL_PROGRAM_DEVICES, cl::vector<Device>) \
F(cl_program_info, CL_PROGRAM_SOURCE, string) \
F(cl_program_info, CL_PROGRAM_BINARY_SIZES, cl::vector<size_type>) \
F(cl_program_info, CL_PROGRAM_BINARIES, cl::vector<cl::vector<unsigned char>>) \
\
F(cl_program_build_info, CL_PROGRAM_BUILD_STATUS, cl_build_status) \
F(cl_program_build_info, CL_PROGRAM_BUILD_OPTIONS, string) \
F(cl_program_build_info, CL_PROGRAM_BUILD_LOG, string) \
\
F(cl_kernel_info, CL_KERNEL_FUNCTION_NAME, string) \
F(cl_kernel_info, CL_KERNEL_NUM_ARGS, cl_uint) \
F(cl_kernel_info, CL_KERNEL_REFERENCE_COUNT, cl_uint) \
F(cl_kernel_info, CL_KERNEL_CONTEXT, cl::Context) \
F(cl_kernel_info, CL_KERNEL_PROGRAM, cl::Program) \
\
F(cl_kernel_work_group_info, CL_KERNEL_WORK_GROUP_SIZE, size_type) \
F(cl_kernel_work_group_info, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, cl::detail::size_t_array) \
F(cl_kernel_work_group_info, CL_KERNEL_LOCAL_MEM_SIZE, cl_ulong) \
\
F(cl_command_queue_info, CL_QUEUE_CONTEXT, cl::Context) \
F(cl_command_queue_info, CL_QUEUE_DEVICE, cl::Device) \
F(cl_command_queue_info, CL_QUEUE_REFERENCE_COUNT, cl_uint) \
F(cl_command_queue_info, CL_QUEUE_PROPERTIES, cl_command_queue_properties)


#define CL_HPP_PARAM_NAME_INFO_1_1_(F) \
F(cl_context_info, CL_CONTEXT_NUM_DEVICES, cl_uint)\
F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint) \
F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl_uint) \
F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
\
F(cl_mem_info, CL_MEM_ASSOCIATED_MEMOBJECT, cl::Memory) \
F(cl_mem_info, CL_MEM_OFFSET, size_type) \
\
F(cl_kernel_work_group_info, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_type) \
F(cl_kernel_work_group_info, CL_KERNEL_PRIVATE_MEM_SIZE, cl_ulong) \
\
F(cl_event_info, CL_EVENT_CONTEXT, cl::Context)

#define CL_HPP_PARAM_NAME_INFO_1_2_(F) \
F(cl_program_info, CL_PROGRAM_NUM_KERNELS, size_type) \
F(cl_program_info, CL_PROGRAM_KERNEL_NAMES, string) \
\
F(cl_program_build_info, CL_PROGRAM_BINARY_TYPE, cl_program_binary_type) \
\
F(cl_kernel_info, CL_KERNEL_ATTRIBUTES, string) \
\
F(cl_kernel_arg_info, CL_KERNEL_ARG_ADDRESS_QUALIFIER, cl_kernel_arg_address_qualifier) \
F(cl_kernel_arg_info, CL_KERNEL_ARG_ACCESS_QUALIFIER, cl_kernel_arg_access_qualifier) \
F(cl_kernel_arg_info, CL_KERNEL_ARG_TYPE_NAME, string) \
F(cl_kernel_arg_info, CL_KERNEL_ARG_NAME, string) \
F(cl_kernel_arg_info, CL_KERNEL_ARG_TYPE_QUALIFIER, cl_kernel_arg_type_qualifier) \
\
F(cl_kernel_work_group_info, CL_KERNEL_GLOBAL_WORK_SIZE, cl::detail::size_t_array) \
\
F(cl_device_info, CL_DEVICE_LINKER_AVAILABLE, cl_bool) \
F(cl_device_info, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, size_type) \
F(cl_device_info, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, size_type) \
F(cl_device_info, CL_DEVICE_PARENT_DEVICE, cl::Device) \
F(cl_device_info, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, cl_uint) \
F(cl_device_info, CL_DEVICE_PARTITION_PROPERTIES, cl::vector<cl_device_partition_property>) \
F(cl_device_info, CL_DEVICE_PARTITION_TYPE, cl::vector<cl_device_partition_property>)  \
F(cl_device_info, CL_DEVICE_REFERENCE_COUNT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, cl_bool) \
F(cl_device_info, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, cl_device_affinity_domain) \
F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS, string) \
F(cl_device_info, CL_DEVICE_PRINTF_BUFFER_SIZE, size_type) \
\
F(cl_image_info, CL_IMAGE_ARRAY_SIZE, size_type) \
F(cl_image_info, CL_IMAGE_NUM_MIP_LEVELS, cl_uint) \
F(cl_image_info, CL_IMAGE_NUM_SAMPLES, cl_uint)

#define CL_HPP_PARAM_NAME_INFO_2_0_(F) \
F(cl_device_info, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, cl_command_queue_properties) \
F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, cl_command_queue_properties) \
F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, cl_uint) \
F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_ON_DEVICE_QUEUES, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_ON_DEVICE_EVENTS, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_PIPE_ARGS, cl_uint) \
F(cl_device_info, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, cl_uint) \
F(cl_device_info, CL_DEVICE_PIPE_MAX_PACKET_SIZE, cl_uint) \
F(cl_device_info, CL_DEVICE_SVM_CAPABILITIES, cl_device_svm_capabilities) \
F(cl_device_info, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, cl_uint) \
F(cl_device_info, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, cl_uint) \
F(cl_device_info, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, cl_uint) \
F(cl_device_info, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, cl_uint) \
F(cl_device_info, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, cl_uint ) \
F(cl_device_info, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, size_type ) \
F(cl_device_info, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, size_type ) \
F(cl_profiling_info, CL_PROFILING_COMMAND_COMPLETE, cl_ulong) \
F(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM, cl_bool) \
F(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_SVM_PTRS, void**) \
F(cl_command_queue_info, CL_QUEUE_SIZE, cl_uint) \
F(cl_mem_info, CL_MEM_USES_SVM_POINTER, cl_bool) \
F(cl_program_build_info, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, size_type) \
F(cl_pipe_info, CL_PIPE_PACKET_SIZE, cl_uint) \
F(cl_pipe_info, CL_PIPE_MAX_PACKETS, cl_uint)

#define CL_HPP_PARAM_NAME_INFO_SUBGROUP_KHR_(F) \
F(cl_kernel_sub_group_info, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR, size_type) \
F(cl_kernel_sub_group_info, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR, size_type)

#define CL_HPP_PARAM_NAME_INFO_IL_KHR_(F) \
F(cl_device_info, CL_DEVICE_IL_VERSION_KHR, string) \
F(cl_program_info, CL_PROGRAM_IL_KHR, cl::vector<unsigned char>)

#define CL_HPP_PARAM_NAME_INFO_2_1_(F) \
F(cl_platform_info, CL_PLATFORM_HOST_TIMER_RESOLUTION, cl_ulong) \
F(cl_program_info, CL_PROGRAM_IL, cl::vector<unsigned char>) \
F(cl_device_info, CL_DEVICE_MAX_NUM_SUB_GROUPS, cl_uint) \
F(cl_device_info, CL_DEVICE_IL_VERSION, string) \
F(cl_device_info, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, cl_bool) \
F(cl_command_queue_info, CL_QUEUE_DEVICE_DEFAULT, cl::DeviceCommandQueue) \
F(cl_kernel_sub_group_info, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, size_type) \
F(cl_kernel_sub_group_info, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, size_type) \
F(cl_kernel_sub_group_info, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, cl::detail::size_t_array) \
F(cl_kernel_sub_group_info, CL_KERNEL_MAX_NUM_SUB_GROUPS, size_type) \
F(cl_kernel_sub_group_info, CL_KERNEL_COMPILE_NUM_SUB_GROUPS, size_type)

#define CL_HPP_PARAM_NAME_INFO_2_2_(F) \
F(cl_program_info, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT, cl_bool) \
F(cl_program_info, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT, cl_bool)

#define CL_HPP_PARAM_NAME_DEVICE_FISSION_(F) \
F(cl_device_info, CL_DEVICE_PARENT_DEVICE_EXT, cl_device_id) \
F(cl_device_info, CL_DEVICE_PARTITION_TYPES_EXT, cl::vector<cl_device_partition_property_ext>) \
F(cl_device_info, CL_DEVICE_AFFINITY_DOMAINS_EXT, cl::vector<cl_device_partition_property_ext>) \
F(cl_device_info, CL_DEVICE_REFERENCE_COUNT_EXT , cl_uint) \
F(cl_device_info, CL_DEVICE_PARTITION_STYLE_EXT, cl::vector<cl_device_partition_property_ext>)

#define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_CL3_SHARED_(F) \
F(cl_platform_info, CL_PLATFORM_NUMERIC_VERSION_KHR, cl_version_khr) \
F(cl_platform_info, CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
\
F(cl_device_info, CL_DEVICE_NUMERIC_VERSION_KHR, cl_version_khr) \
F(cl_device_info, CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
F(cl_device_info, CL_DEVICE_ILS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>)

#define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_KHRONLY_(F) \
F(cl_device_info, CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, cl_version_khr)

#define CL_HPP_PARAM_NAME_INFO_3_0_(F) \
F(cl_platform_info, CL_PLATFORM_NUMERIC_VERSION, cl_version) \
F(cl_platform_info, CL_PLATFORM_EXTENSIONS_WITH_VERSION, cl::vector<cl_name_version>) \
\
F(cl_device_info, CL_DEVICE_NUMERIC_VERSION, cl_version) \
F(cl_device_info, CL_DEVICE_EXTENSIONS_WITH_VERSION, cl::vector<cl_name_version>) \
F(cl_device_info, CL_DEVICE_ILS_WITH_VERSION, cl::vector<cl_name_version>) \
F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION, cl::vector<cl_name_version>) \
F(cl_device_info, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, cl_device_atomic_capabilities) \
F(cl_device_info, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES, cl_device_atomic_capabilities) \
F(cl_device_info, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_OPENCL_C_ALL_VERSIONS, cl::vector<cl_name_version>) \
F(cl_device_info, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_type) \
F(cl_device_info, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_OPENCL_C_FEATURES, cl::vector<cl_name_version>) \
F(cl_device_info, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES, cl_device_device_enqueue_capabilities) \
F(cl_device_info, CL_DEVICE_PIPE_SUPPORT, cl_bool) \
F(cl_device_info, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED, string) \
\
F(cl_command_queue_info, CL_QUEUE_PROPERTIES_ARRAY, cl::vector<cl_queue_properties>) \
F(cl_mem_info, CL_MEM_PROPERTIES, cl::vector<cl_mem_properties>) \
F(cl_pipe_info, CL_PIPE_PROPERTIES, cl::vector<cl_pipe_properties>) \
F(cl_sampler_info, CL_SAMPLER_PROPERTIES, cl::vector<cl_sampler_properties>)

template <typename enum_type, cl_int Name>
struct param_traits {};

#define CL_HPP_DECLARE_PARAM_TRAITS_(token, param_name, T) \
struct token;                                        \
template<>                                           \
struct param_traits<detail:: token,param_name>       \
{                                                    \
enum { value = param_name };                     \
typedef T param_type;                            \
};

CL_HPP_PARAM_NAME_INFO_1_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
CL_HPP_PARAM_NAME_INFO_1_1_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
CL_HPP_PARAM_NAME_INFO_1_2_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
CL_HPP_PARAM_NAME_INFO_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
CL_HPP_PARAM_NAME_INFO_2_1_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 220
CL_HPP_PARAM_NAME_INFO_2_2_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 300
CL_HPP_PARAM_NAME_INFO_3_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 

#if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
CL_HPP_PARAM_NAME_INFO_SUBGROUP_KHR_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 

#if defined(CL_HPP_USE_IL_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
CL_HPP_PARAM_NAME_INFO_IL_KHR_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 


#define CL_HPP_PARAM_NAME_INFO_1_0_DEPRECATED_IN_2_0_(F) \
F(cl_device_info, CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties)

#define CL_HPP_PARAM_NAME_INFO_1_1_DEPRECATED_IN_2_0_(F) \
F(cl_device_info, CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool)

#define CL_HPP_PARAM_NAME_INFO_1_2_DEPRECATED_IN_2_0_(F) \
F(cl_image_info, CL_IMAGE_BUFFER, cl::Buffer)

#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_0_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_1_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_2_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 

#if defined(CL_HPP_USE_CL_DEVICE_FISSION)
CL_HPP_PARAM_NAME_DEVICE_FISSION_(CL_HPP_DECLARE_PARAM_TRAITS_);
#endif 

#if defined(cl_khr_extended_versioning)
#if CL_HPP_TARGET_OPENCL_VERSION < 300
CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_CL3_SHARED_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 
CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_KHRONLY_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif 

#if defined(cl_khr_device_uuid)
using uuid_array = array<cl_uchar, CL_UUID_SIZE_KHR>;
using luid_array = array<cl_uchar, CL_LUID_SIZE_KHR>;
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_UUID_KHR, uuid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DRIVER_UUID_KHR, uuid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LUID_VALID_KHR, cl_bool)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LUID_KHR, luid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NODE_MASK_KHR, cl_uint)
#endif

#if defined(cl_khr_pci_bus_info)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_PCI_BUS_INFO_KHR, cl_device_pci_bus_info_khr)
#endif

#if defined(cl_khr_integer_dot_product)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR, cl_device_integer_dot_product_capabilities_khr)
#if defined(CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR, cl_device_integer_dot_product_acceleration_properties_khr)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR, cl_device_integer_dot_product_acceleration_properties_khr)
#endif 
#endif 

#ifdef CL_PLATFORM_ICD_SUFFIX_KHR
CL_HPP_DECLARE_PARAM_TRAITS_(cl_platform_info, CL_PLATFORM_ICD_SUFFIX_KHR, string)
#endif

#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_PROFILING_TIMER_OFFSET_AMD, cl_ulong)
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, vector<size_type>)
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_WAVEFRONT_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LOCAL_MEM_BANKS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_BOARD_NAME_AMD, string)
#endif

#ifdef CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM, cl_ulong)
#endif
#ifdef CL_DEVICE_JOB_SLOTS_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_JOB_SLOTS_ARM, cl_uint)
#endif
#ifdef CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM, cl_bitfield)
#endif
#ifdef CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM, vector<cl_uint>)
#endif
#ifdef CL_DEVICE_MAX_WARP_COUNT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_MAX_WARP_COUNT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_MAX_WARP_COUNT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_info, CL_KERNEL_MAX_WARP_COUNT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM, cl_int)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WARP_COUNT_LIMIT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WARP_COUNT_LIMIT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_COMPUTE_UNIT_MAX_QUEUED_BATCHES_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_COMPUTE_UNIT_MAX_QUEUED_BATCHES_ARM, cl_uint)
#endif

#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, cl_uint)
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, cl_uint)
#endif
#ifdef CL_DEVICE_REGISTERS_PER_BLOCK_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_REGISTERS_PER_BLOCK_NV, cl_uint)
#endif
#ifdef CL_DEVICE_WARP_SIZE_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_WARP_SIZE_NV, cl_uint)
#endif
#ifdef CL_DEVICE_GPU_OVERLAP_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
#endif
#ifdef CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, cl_bool)
#endif
#ifdef CL_DEVICE_INTEGRATED_MEMORY_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGRATED_MEMORY_NV, cl_bool)
#endif


template <typename Func, typename T>
inline cl_int
getInfo(Func f, cl_uint name, T* param)
{
return getInfoHelper(f, name, param, 0);
}

template <typename Func, typename Arg0>
struct GetInfoFunctor0
{
Func f_; const Arg0& arg0_;
cl_int operator ()(
cl_uint param, size_type size, void* value, size_type* size_ret)
{ return f_(arg0_, param, size, value, size_ret); }
};

template <typename Func, typename Arg0, typename Arg1>
struct GetInfoFunctor1
{
Func f_; const Arg0& arg0_; const Arg1& arg1_;
cl_int operator ()(
cl_uint param, size_type size, void* value, size_type* size_ret)
{ return f_(arg0_, arg1_, param, size, value, size_ret); }
};

template <typename Func, typename Arg0, typename T>
inline cl_int
getInfo(Func f, const Arg0& arg0, cl_uint name, T* param)
{
GetInfoFunctor0<Func, Arg0> f0 = { f, arg0 };
return getInfoHelper(f0, name, param, 0);
}

template <typename Func, typename Arg0, typename Arg1, typename T>
inline cl_int
getInfo(Func f, const Arg0& arg0, const Arg1& arg1, cl_uint name, T* param)
{
GetInfoFunctor1<Func, Arg0, Arg1> f0 = { f, arg0, arg1 };
return getInfoHelper(f0, name, param, 0);
}


template<typename T>
struct ReferenceHandler
{ };

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

template <>
struct ReferenceHandler<cl_device_id>
{

static cl_int retain(cl_device_id device)
{ return ::clRetainDevice(device); }

static cl_int release(cl_device_id device)
{ return ::clReleaseDevice(device); }
};
#else 

template <>
struct ReferenceHandler<cl_device_id>
{
static cl_int retain(cl_device_id)
{ return CL_SUCCESS; }
static cl_int release(cl_device_id)
{ return CL_SUCCESS; }
};
#endif 

template <>
struct ReferenceHandler<cl_platform_id>
{
static cl_int retain(cl_platform_id)
{ return CL_SUCCESS; }
static cl_int release(cl_platform_id)
{ return CL_SUCCESS; }
};

template <>
struct ReferenceHandler<cl_context>
{
static cl_int retain(cl_context context)
{ return ::clRetainContext(context); }
static cl_int release(cl_context context)
{ return ::clReleaseContext(context); }
};

template <>
struct ReferenceHandler<cl_command_queue>
{
static cl_int retain(cl_command_queue queue)
{ return ::clRetainCommandQueue(queue); }
static cl_int release(cl_command_queue queue)
{ return ::clReleaseCommandQueue(queue); }
};

template <>
struct ReferenceHandler<cl_mem>
{
static cl_int retain(cl_mem memory)
{ return ::clRetainMemObject(memory); }
static cl_int release(cl_mem memory)
{ return ::clReleaseMemObject(memory); }
};

template <>
struct ReferenceHandler<cl_sampler>
{
static cl_int retain(cl_sampler sampler)
{ return ::clRetainSampler(sampler); }
static cl_int release(cl_sampler sampler)
{ return ::clReleaseSampler(sampler); }
};

template <>
struct ReferenceHandler<cl_program>
{
static cl_int retain(cl_program program)
{ return ::clRetainProgram(program); }
static cl_int release(cl_program program)
{ return ::clReleaseProgram(program); }
};

template <>
struct ReferenceHandler<cl_kernel>
{
static cl_int retain(cl_kernel kernel)
{ return ::clRetainKernel(kernel); }
static cl_int release(cl_kernel kernel)
{ return ::clReleaseKernel(kernel); }
};

template <>
struct ReferenceHandler<cl_event>
{
static cl_int retain(cl_event event)
{ return ::clRetainEvent(event); }
static cl_int release(cl_event event)
{ return ::clReleaseEvent(event); }
};


#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
static cl_uint getVersion(const vector<char> &versionInfo)
{
int highVersion = 0;
int lowVersion = 0;
int index = 7;
while(versionInfo[index] != '.' ) {
highVersion *= 10;
highVersion += versionInfo[index]-'0';
++index;
}
++index;
while(versionInfo[index] != ' ' &&  versionInfo[index] != '\0') {
lowVersion *= 10;
lowVersion += versionInfo[index]-'0';
++index;
}
return (highVersion << 16) | lowVersion;
}

static cl_uint getPlatformVersion(cl_platform_id platform)
{
size_type size = 0;
clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &size);

vector<char> versionInfo(size);
clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, versionInfo.data(), &size);
return getVersion(versionInfo);
}

static cl_uint getDevicePlatformVersion(cl_device_id device)
{
cl_platform_id platform;
clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL);
return getPlatformVersion(platform);
}

static cl_uint getContextPlatformVersion(cl_context context)
{
size_type size = 0;
clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
if (size == 0)
return 0;
vector<cl_device_id> devices(size/sizeof(cl_device_id));
clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices.data(), NULL);
return getDevicePlatformVersion(devices[0]);
}
#endif 

template <typename T>
class Wrapper
{
public:
typedef T cl_type;

protected:
cl_type object_;

public:
Wrapper() : object_(NULL) { }

Wrapper(const cl_type &obj, bool retainObject) : object_(obj) 
{
if (retainObject) { 
detail::errHandler(retain(), __RETAIN_ERR); 
}
}

~Wrapper()
{
if (object_ != NULL) { release(); }
}

Wrapper(const Wrapper<cl_type>& rhs)
{
object_ = rhs.object_;
detail::errHandler(retain(), __RETAIN_ERR);
}

Wrapper(Wrapper<cl_type>&& rhs) CL_HPP_NOEXCEPT_
{
object_ = rhs.object_;
rhs.object_ = NULL;
}

Wrapper<cl_type>& operator = (const Wrapper<cl_type>& rhs)
{
if (this != &rhs) {
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs.object_;
detail::errHandler(retain(), __RETAIN_ERR);
}
return *this;
}

Wrapper<cl_type>& operator = (Wrapper<cl_type>&& rhs)
{
if (this != &rhs) {
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs.object_;
rhs.object_ = NULL;
}
return *this;
}

Wrapper<cl_type>& operator = (const cl_type &rhs)
{
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs;
return *this;
}

const cl_type& operator ()() const { return object_; }

cl_type& operator ()() { return object_; }

cl_type get() const { return object_; }

protected:
template<typename Func, typename U>
friend inline cl_int getInfoHelper(Func, cl_uint, U*, int, typename U::cl_type);

cl_int retain() const
{
if (object_ != nullptr) {
return ReferenceHandler<cl_type>::retain(object_);
}
else {
return CL_SUCCESS;
}
}

cl_int release() const
{
if (object_ != nullptr) {
return ReferenceHandler<cl_type>::release(object_);
}
else {
return CL_SUCCESS;
}
}
};

template <>
class Wrapper<cl_device_id>
{
public:
typedef cl_device_id cl_type;

protected:
cl_type object_;
bool referenceCountable_;

static bool isReferenceCountable(cl_device_id device)
{
bool retVal = false;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
if (device != NULL) {
int version = getDevicePlatformVersion(device);
if(version > ((1 << 16) + 1)) {
retVal = true;
}
}
#else 
retVal = true;
#endif 
#endif 
(void)device;
return retVal;
}

public:
Wrapper() : object_(NULL), referenceCountable_(false) 
{ 
}

Wrapper(const cl_type &obj, bool retainObject) : 
object_(obj), 
referenceCountable_(false) 
{
referenceCountable_ = isReferenceCountable(obj); 

if (retainObject) {
detail::errHandler(retain(), __RETAIN_ERR);
}
}

~Wrapper()
{
release();
}

Wrapper(const Wrapper<cl_type>& rhs)
{
object_ = rhs.object_;
referenceCountable_ = isReferenceCountable(object_); 
detail::errHandler(retain(), __RETAIN_ERR);
}

Wrapper(Wrapper<cl_type>&& rhs) CL_HPP_NOEXCEPT_
{
object_ = rhs.object_;
referenceCountable_ = rhs.referenceCountable_;
rhs.object_ = NULL;
rhs.referenceCountable_ = false;
}

Wrapper<cl_type>& operator = (const Wrapper<cl_type>& rhs)
{
if (this != &rhs) {
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs.object_;
referenceCountable_ = rhs.referenceCountable_;
detail::errHandler(retain(), __RETAIN_ERR);
}
return *this;
}

Wrapper<cl_type>& operator = (Wrapper<cl_type>&& rhs)
{
if (this != &rhs) {
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs.object_;
referenceCountable_ = rhs.referenceCountable_;
rhs.object_ = NULL;
rhs.referenceCountable_ = false;
}
return *this;
}

Wrapper<cl_type>& operator = (const cl_type &rhs)
{
detail::errHandler(release(), __RELEASE_ERR);
object_ = rhs;
referenceCountable_ = isReferenceCountable(object_); 
return *this;
}

const cl_type& operator ()() const { return object_; }

cl_type& operator ()() { return object_; }

cl_type get() const { return object_; }

protected:
template<typename Func, typename U>
friend inline cl_int getInfoHelper(Func, cl_uint, U*, int, typename U::cl_type);

template<typename Func, typename U>
friend inline cl_int getInfoHelper(Func, cl_uint, vector<U>*, int, typename U::cl_type);

cl_int retain() const
{
if( object_ != nullptr && referenceCountable_ ) {
return ReferenceHandler<cl_type>::retain(object_);
}
else {
return CL_SUCCESS;
}
}

cl_int release() const
{
if (object_ != nullptr && referenceCountable_) {
return ReferenceHandler<cl_type>::release(object_);
}
else {
return CL_SUCCESS;
}
}
};

template <typename T>
inline bool operator==(const Wrapper<T> &lhs, const Wrapper<T> &rhs)
{
return lhs() == rhs();
}

template <typename T>
inline bool operator!=(const Wrapper<T> &lhs, const Wrapper<T> &rhs)
{
return !operator==(lhs, rhs);
}

} 






struct ImageFormat : public cl_image_format
{
ImageFormat(){}

ImageFormat(cl_channel_order order, cl_channel_type type)
{
image_channel_order = order;
image_channel_data_type = type;
}

ImageFormat(const ImageFormat &other) { *this = other; }

ImageFormat& operator = (const ImageFormat& rhs)
{
if (this != &rhs) {
this->image_channel_data_type = rhs.image_channel_data_type;
this->image_channel_order     = rhs.image_channel_order;
}
return *this;
}
};


class Device : public detail::Wrapper<cl_device_id>
{
private:
static std::once_flag default_initialized_;
static Device default_;
static cl_int default_error_;


static void makeDefault();


static void makeDefaultProvided(const Device &p) {
default_ = p;
}

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE

static void unitTestClearDefault() {
default_ = Device();
}
#endif 

Device() : detail::Wrapper<cl_type>() { }


explicit Device(const cl_device_id &device, bool retainObject = false) : 
detail::Wrapper<cl_type>(device, retainObject) { }


static Device getDefault(
cl_int *errResult = NULL)
{
std::call_once(default_initialized_, makeDefault);
detail::errHandler(default_error_);
if (errResult != NULL) {
*errResult = default_error_;
}
return default_;
}


static Device setDefault(const Device &default_device)
{
std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_device));
detail::errHandler(default_error_);
return default_;
}


Device& operator = (const cl_device_id& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


Device(const Device& dev) : detail::Wrapper<cl_type>(dev) {}


Device& operator = (const Device &dev)
{
detail::Wrapper<cl_type>::operator=(dev);
return *this;
}


Device(Device&& dev) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(dev)) {}


Device& operator = (Device &&dev)
{
detail::Wrapper<cl_type>::operator=(std::move(dev));
return *this;
}

template <typename T>
cl_int getInfo(cl_device_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetDeviceInfo, object_, name, param),
__GET_DEVICE_INFO_ERR);
}

template <cl_device_info name> typename
detail::param_traits<detail::cl_device_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_device_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 210

cl_ulong getHostTimer(cl_int *error = nullptr)
{
cl_ulong retVal = 0;
cl_int err = 
clGetHostTimer(this->get(), &retVal);
detail::errHandler(
err,
__GET_HOST_TIMER_ERR);
if (error) {
*error = err;
}
return retVal;
}


std::pair<cl_ulong, cl_ulong> getDeviceAndHostTimer(cl_int *error = nullptr)
{
std::pair<cl_ulong, cl_ulong> retVal;
cl_int err =
clGetDeviceAndHostTimer(this->get(), &(retVal.first), &(retVal.second));
detail::errHandler(
err,
__GET_DEVICE_AND_HOST_TIMER_ERR);
if (error) {
*error = err;
}
return retVal;
}
#endif 


#if CL_HPP_TARGET_OPENCL_VERSION >= 120
cl_int createSubDevices(
const cl_device_partition_property * properties,
vector<Device>* devices)
{
cl_uint n = 0;
cl_int err = clCreateSubDevices(object_, properties, 0, NULL, &n);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
}

vector<cl_device_id> ids(n);
err = clCreateSubDevices(object_, properties, n, ids.data(), NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
}

if (devices) {
devices->resize(ids.size());

for (size_type i = 0; i < ids.size(); i++) {
(*devices)[i] = Device(ids[i], false);
}
}

return CL_SUCCESS;
}
#elif defined(CL_HPP_USE_CL_DEVICE_FISSION)


cl_int createSubDevices(
const cl_device_partition_property_ext * properties,
vector<Device>* devices)
{
typedef CL_API_ENTRY cl_int 
( CL_API_CALL * PFN_clCreateSubDevicesEXT)(
cl_device_id ,
const cl_device_partition_property_ext * ,
cl_uint ,
cl_device_id * ,
cl_uint *  ) CL_API_SUFFIX__VERSION_1_1;

static PFN_clCreateSubDevicesEXT pfn_clCreateSubDevicesEXT = NULL;
CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateSubDevicesEXT);

cl_uint n = 0;
cl_int err = pfn_clCreateSubDevicesEXT(object_, properties, 0, NULL, &n);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
}

vector<cl_device_id> ids(n);
err = pfn_clCreateSubDevicesEXT(object_, properties, n, ids.data(), NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
}
if (devices) {
devices->resize(ids.size());

for (size_type i = 0; i < ids.size(); i++) {
(*devices)[i] = Device(ids[i], false);
}
}
return CL_SUCCESS;
}
#endif 
};

using BuildLogType = vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, CL_PROGRAM_BUILD_LOG>::param_type>>;
#if defined(CL_HPP_ENABLE_EXCEPTIONS)

class BuildError : public Error
{
private:
BuildLogType buildLogs;
public:
BuildError(cl_int err, const char * errStr, const BuildLogType &vec) : Error(err, errStr), buildLogs(vec)
{
}

BuildLogType getBuildLog() const
{
return buildLogs;
}
};
namespace detail {
static inline cl_int buildErrHandler(
cl_int err,
const char * errStr,
const BuildLogType &buildLogs)
{
if (err != CL_SUCCESS) {
throw BuildError(err, errStr, buildLogs);
}
return err;
}
} 

#else
namespace detail {
static inline cl_int buildErrHandler(
cl_int err,
const char * errStr,
const BuildLogType &buildLogs)
{
(void)buildLogs; 
(void)errStr;
return err;
}
} 
#endif 

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Device::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Device Device::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Device::default_error_ = CL_SUCCESS;


class Platform : public detail::Wrapper<cl_platform_id>
{
private:
static std::once_flag default_initialized_;
static Platform default_;
static cl_int default_error_;


static void makeDefault() {

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
try
#endif
{
cl_uint n = 0;

cl_int err = ::clGetPlatformIDs(0, NULL, &n);
if (err != CL_SUCCESS) {
default_error_ = err;
return;
}
if (n == 0) {
default_error_ = CL_INVALID_PLATFORM;
return;
}

vector<cl_platform_id> ids(n);
err = ::clGetPlatformIDs(n, ids.data(), NULL);
if (err != CL_SUCCESS) {
default_error_ = err;
return;
}

default_ = Platform(ids[0]);
}
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
catch (cl::Error &e) {
default_error_ = e.err();
}
#endif
}


static void makeDefaultProvided(const Platform &p) {
default_ = p;
}

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE

static void unitTestClearDefault() {
default_ = Platform();
}
#endif 

Platform() : detail::Wrapper<cl_type>()  { }


explicit Platform(const cl_platform_id &platform, bool retainObject = false) : 
detail::Wrapper<cl_type>(platform, retainObject) { }


Platform& operator = (const cl_platform_id& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}

static Platform getDefault(
cl_int *errResult = NULL)
{
std::call_once(default_initialized_, makeDefault);
detail::errHandler(default_error_);
if (errResult != NULL) {
*errResult = default_error_;
}
return default_;
}


static Platform setDefault(const Platform &default_platform)
{
std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_platform));
detail::errHandler(default_error_);
return default_;
}

template <typename T>
cl_int getInfo(cl_platform_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetPlatformInfo, object_, name, param),
__GET_PLATFORM_INFO_ERR);
}

template <cl_platform_info name> typename
detail::param_traits<detail::cl_platform_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_platform_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


cl_int getDevices(
cl_device_type type,
vector<Device>* devices) const
{
cl_uint n = 0;
if( devices == NULL ) {
return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_DEVICE_IDS_ERR);
}
cl_int err = ::clGetDeviceIDs(object_, type, 0, NULL, &n);
if (err != CL_SUCCESS  && err != CL_DEVICE_NOT_FOUND) {
return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
}

vector<cl_device_id> ids(n);
if (n>0) {
err = ::clGetDeviceIDs(object_, type, n, ids.data(), NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
}
}

if (devices) {
devices->resize(ids.size());

for (size_type i = 0; i < ids.size(); i++) {
(*devices)[i] = Device(ids[i], true);
}
}
return CL_SUCCESS;
}

#if defined(CL_HPP_USE_DX_INTEROP)

cl_int getDevices(
cl_d3d10_device_source_khr d3d_device_source,
void *                     d3d_object,
cl_d3d10_device_set_khr    d3d_device_set,
vector<Device>* devices) const
{
typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clGetDeviceIDsFromD3D10KHR)(
cl_platform_id platform, 
cl_d3d10_device_source_khr d3d_device_source, 
void * d3d_object,
cl_d3d10_device_set_khr d3d_device_set,
cl_uint num_entries,
cl_device_id * devices,
cl_uint* num_devices);

if( devices == NULL ) {
return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_DEVICE_IDS_ERR);
}

static PFN_clGetDeviceIDsFromD3D10KHR pfn_clGetDeviceIDsFromD3D10KHR = NULL;
CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(object_, clGetDeviceIDsFromD3D10KHR);

cl_uint n = 0;
cl_int err = pfn_clGetDeviceIDsFromD3D10KHR(
object_, 
d3d_device_source, 
d3d_object,
d3d_device_set, 
0, 
NULL, 
&n);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
}

vector<cl_device_id> ids(n);
err = pfn_clGetDeviceIDsFromD3D10KHR(
object_, 
d3d_device_source, 
d3d_object,
d3d_device_set,
n, 
ids.data(), 
NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
}

if (devices) {
devices->resize(ids.size());

for (size_type i = 0; i < ids.size(); i++) {
(*devices)[i] = Device(ids[i], true);
}
}
return CL_SUCCESS;
}
#endif


static cl_int get(
vector<Platform>* platforms)
{
cl_uint n = 0;

if( platforms == NULL ) {
return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_PLATFORM_IDS_ERR);
}

cl_int err = ::clGetPlatformIDs(0, NULL, &n);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_PLATFORM_IDS_ERR);
}

vector<cl_platform_id> ids(n);
err = ::clGetPlatformIDs(n, ids.data(), NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_PLATFORM_IDS_ERR);
}

if (platforms) {
platforms->resize(ids.size());

for (size_type i = 0; i < ids.size(); i++) {
(*platforms)[i] = Platform(ids[i]);
}
}
return CL_SUCCESS;
}


static cl_int get(
Platform * platform)
{
cl_int err;
Platform default_platform = Platform::getDefault(&err);
if (platform) {
*platform = default_platform;
}
return err;
}


static Platform get(
cl_int * errResult = NULL)
{
cl_int err;
Platform default_platform = Platform::getDefault(&err);
if (errResult) {
*errResult = err;
}
return default_platform;
}    

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
cl_int
unloadCompiler()
{
return ::clUnloadPlatformCompiler(object_);
}
#endif 
}; 

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Platform::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Platform Platform::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Platform::default_error_ = CL_SUCCESS;



#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

inline CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int
UnloadCompiler() CL_API_SUFFIX__VERSION_1_1_DEPRECATED;
inline cl_int
UnloadCompiler()
{
return ::clUnloadCompiler();
}
#endif 


class Context 
: public detail::Wrapper<cl_context>
{
private:
static std::once_flag default_initialized_;
static Context default_;
static cl_int default_error_;


static void makeDefault() {

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
try
#endif
{
#if !defined(__APPLE__) && !defined(__MACOS)
const Platform &p = Platform::getDefault();
cl_platform_id defaultPlatform = p();
cl_context_properties properties[3] = {
CL_CONTEXT_PLATFORM, (cl_context_properties)defaultPlatform, 0
};
#else 
cl_context_properties *properties = nullptr;
#endif 

default_ = Context(
CL_DEVICE_TYPE_DEFAULT,
properties,
NULL,
NULL,
&default_error_);
}
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
catch (cl::Error &e) {
default_error_ = e.err();
}
#endif
}



static void makeDefaultProvided(const Context &c) {
default_ = c;
}

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE

static void unitTestClearDefault() {
default_ = Context();
}
#endif 


Context(
const vector<Device>& devices,
const cl_context_properties* properties = NULL,
void (CL_CALLBACK * notifyFptr)(
const char *,
const void *,
size_type,
void *) = NULL,
void* data = NULL,
cl_int* err = NULL)
{
cl_int error;

size_type numDevices = devices.size();
vector<cl_device_id> deviceIDs(numDevices);

for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
deviceIDs[deviceIndex] = (devices[deviceIndex])();
}

object_ = ::clCreateContext(
properties, (cl_uint) numDevices,
deviceIDs.data(),
notifyFptr, data, &error);

detail::errHandler(error, __CREATE_CONTEXT_ERR);
if (err != NULL) {
*err = error;
}
}


Context(
const Device& device,
const cl_context_properties* properties = NULL,
void (CL_CALLBACK * notifyFptr)(
const char *,
const void *,
size_type,
void *) = NULL,
void* data = NULL,
cl_int* err = NULL)
{
cl_int error;

cl_device_id deviceID = device();

object_ = ::clCreateContext(
properties, 1,
&deviceID,
notifyFptr, data, &error);

detail::errHandler(error, __CREATE_CONTEXT_ERR);
if (err != NULL) {
*err = error;
}
}


Context(
cl_device_type type,
const cl_context_properties* properties = NULL,
void (CL_CALLBACK * notifyFptr)(
const char *,
const void *,
size_type,
void *) = NULL,
void* data = NULL,
cl_int* err = NULL)
{
cl_int error;

#if !defined(__APPLE__) && !defined(__MACOS)
cl_context_properties prop[4] = {CL_CONTEXT_PLATFORM, 0, 0, 0 };

if (properties == NULL) {
vector<Platform> platforms;
error = Platform::get(&platforms);
if (error != CL_SUCCESS) {
detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
if (err != NULL) {
*err = error;
}
return;
}

cl_context_properties platform_id = 0;
for (unsigned int i = 0; i < platforms.size(); i++) {

vector<Device> devices;

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
try {
#endif

error = platforms[i].getDevices(type, &devices);

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
} catch (cl::Error& e) {
error = e.err();
}
#endif

if (error != CL_SUCCESS && error != CL_DEVICE_NOT_FOUND) {
detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
if (err != NULL) {
*err = error;
}
}

if (devices.size() > 0) {
platform_id = (cl_context_properties)platforms[i]();
break;
}
}

if (platform_id == 0) {
detail::errHandler(CL_DEVICE_NOT_FOUND, __CREATE_CONTEXT_FROM_TYPE_ERR);
if (err != NULL) {
*err = CL_DEVICE_NOT_FOUND;
}
return;
}

prop[1] = platform_id;
properties = &prop[0];
}
#endif
object_ = ::clCreateContextFromType(
properties, type, notifyFptr, data, &error);

detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
if (err != NULL) {
*err = error;
}
}


Context(const Context& ctx) : detail::Wrapper<cl_type>(ctx) {}


Context& operator = (const Context &ctx)
{
detail::Wrapper<cl_type>::operator=(ctx);
return *this;
}


Context(Context&& ctx) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(ctx)) {}


Context& operator = (Context &&ctx)
{
detail::Wrapper<cl_type>::operator=(std::move(ctx));
return *this;
}



static Context getDefault(cl_int * err = NULL) 
{
std::call_once(default_initialized_, makeDefault);
detail::errHandler(default_error_);
if (err != NULL) {
*err = default_error_;
}
return default_;
}


static Context setDefault(const Context &default_context)
{
std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_context));
detail::errHandler(default_error_);
return default_;
}

Context() : detail::Wrapper<cl_type>() { }


explicit Context(const cl_context& context, bool retainObject = false) : 
detail::Wrapper<cl_type>(context, retainObject) { }


Context& operator = (const cl_context& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}

template <typename T>
cl_int getInfo(cl_context_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetContextInfo, object_, name, param),
__GET_CONTEXT_INFO_ERR);
}

template <cl_context_info name> typename
detail::param_traits<detail::cl_context_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_context_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


cl_int getSupportedImageFormats(
cl_mem_flags flags,
cl_mem_object_type type,
vector<ImageFormat>* formats) const
{
cl_uint numEntries;

if (!formats) {
return CL_SUCCESS;
}

cl_int err = ::clGetSupportedImageFormats(
object_, 
flags,
type, 
0, 
NULL, 
&numEntries);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_SUPPORTED_IMAGE_FORMATS_ERR);
}

if (numEntries > 0) {
vector<ImageFormat> value(numEntries);
err = ::clGetSupportedImageFormats(
object_,
flags,
type,
numEntries,
(cl_image_format*)value.data(),
NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __GET_SUPPORTED_IMAGE_FORMATS_ERR);
}

formats->assign(begin(value), end(value));
}
else {
formats->clear();
}

return CL_SUCCESS;
}
};

inline void Device::makeDefault()
{

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
try
#endif
{
cl_int error = 0;

Context context = Context::getDefault(&error);
detail::errHandler(error, __CREATE_CONTEXT_ERR);

if (error != CL_SUCCESS) {
default_error_ = error;
}
else {
default_ = context.getInfo<CL_CONTEXT_DEVICES>()[0];
default_error_ = CL_SUCCESS;
}
}
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
catch (cl::Error &e) {
default_error_ = e.err();
}
#endif
}

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Context::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Context Context::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Context::default_error_ = CL_SUCCESS;


class Event : public detail::Wrapper<cl_event>
{
public:
Event() : detail::Wrapper<cl_type>() { }


explicit Event(const cl_event& event, bool retainObject = false) : 
detail::Wrapper<cl_type>(event, retainObject) { }


Event& operator = (const cl_event& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}

template <typename T>
cl_int getInfo(cl_event_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetEventInfo, object_, name, param),
__GET_EVENT_INFO_ERR);
}

template <cl_event_info name> typename
detail::param_traits<detail::cl_event_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_event_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

template <typename T>
cl_int getProfilingInfo(cl_profiling_info name, T* param) const
{
return detail::errHandler(detail::getInfo(
&::clGetEventProfilingInfo, object_, name, param),
__GET_EVENT_PROFILE_INFO_ERR);
}

template <cl_profiling_info name> typename
detail::param_traits<detail::cl_profiling_info, name>::param_type
getProfilingInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_profiling_info, name>::param_type param;
cl_int result = getProfilingInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


cl_int wait() const
{
return detail::errHandler(
::clWaitForEvents(1, &object_),
__WAIT_FOR_EVENTS_ERR);
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 110

cl_int setCallback(
cl_int type,
void (CL_CALLBACK * pfn_notify)(cl_event, cl_int, void *),
void * user_data = NULL)
{
return detail::errHandler(
::clSetEventCallback(
object_,
type,
pfn_notify,
user_data), 
__SET_EVENT_CALLBACK_ERR);
}
#endif 


static cl_int
waitForEvents(const vector<Event>& events)
{
return detail::errHandler(
::clWaitForEvents(
(cl_uint) events.size(), (events.size() > 0) ? (cl_event*)&events.front() : NULL),
__WAIT_FOR_EVENTS_ERR);
}
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 110

class UserEvent : public Event
{
public:

UserEvent(
const Context& context,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateUserEvent(
context(),
&error);

detail::errHandler(error, __CREATE_USER_EVENT_ERR);
if (err != NULL) {
*err = error;
}
}

UserEvent() : Event() { }


cl_int setStatus(cl_int status)
{
return detail::errHandler(
::clSetUserEventStatus(object_,status), 
__SET_USER_EVENT_STATUS_ERR);
}
};
#endif 


inline static cl_int
WaitForEvents(const vector<Event>& events)
{
return detail::errHandler(
::clWaitForEvents(
(cl_uint) events.size(), (events.size() > 0) ? (cl_event*)&events.front() : NULL),
__WAIT_FOR_EVENTS_ERR);
}


class Memory : public detail::Wrapper<cl_mem>
{
public:
Memory() : detail::Wrapper<cl_type>() { }


explicit Memory(const cl_mem& memory, bool retainObject) :
detail::Wrapper<cl_type>(memory, retainObject) { }


Memory& operator = (const cl_mem& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


Memory(const Memory& mem) : detail::Wrapper<cl_type>(mem) {}


Memory& operator = (const Memory &mem)
{
detail::Wrapper<cl_type>::operator=(mem);
return *this;
}


Memory(Memory&& mem) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(mem)) {}


Memory& operator = (Memory &&mem)
{
detail::Wrapper<cl_type>::operator=(std::move(mem));
return *this;
}


template <typename T>
cl_int getInfo(cl_mem_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetMemObjectInfo, object_, name, param),
__GET_MEM_OBJECT_INFO_ERR);
}

template <cl_mem_info name> typename
detail::param_traits<detail::cl_mem_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_mem_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 110

cl_int setDestructorCallback(
void (CL_CALLBACK * pfn_notify)(cl_mem, void *),
void * user_data = NULL)
{
return detail::errHandler(
::clSetMemObjectDestructorCallback(
object_,
pfn_notify,
user_data), 
__SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR);
}
#endif 

};

class Buffer;
template< typename IteratorType >
cl_int copy( IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer );
template< typename IteratorType >
cl_int copy( const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator );
template< typename IteratorType >
cl_int copy( const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer );
template< typename IteratorType >
cl_int copy( const CommandQueue &queue, const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator );


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
namespace detail
{
class SVMTraitNull
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return 0;
}
};
} 

template<class Trait = detail::SVMTraitNull>
class SVMTraitReadWrite
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return CL_MEM_READ_WRITE |
Trait::getSVMMemFlags();
}
};

template<class Trait = detail::SVMTraitNull>
class SVMTraitReadOnly
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return CL_MEM_READ_ONLY |
Trait::getSVMMemFlags();
}
};

template<class Trait = detail::SVMTraitNull>
class SVMTraitWriteOnly
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return CL_MEM_WRITE_ONLY |
Trait::getSVMMemFlags();
}
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitCoarse
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return Trait::getSVMMemFlags();
}
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitFine
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return CL_MEM_SVM_FINE_GRAIN_BUFFER |
Trait::getSVMMemFlags();
}
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitAtomic
{
public:
static cl_svm_mem_flags getSVMMemFlags()
{
return
CL_MEM_SVM_FINE_GRAIN_BUFFER |
CL_MEM_SVM_ATOMICS |
Trait::getSVMMemFlags();
}
};

template<typename T>
inline cl_int enqueueMapSVM(
T* ptr,
cl_bool blocking,
cl_map_flags flags,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL);


template<typename T, class SVMTrait>
class SVMAllocator {
private:
Context context_;

public:
typedef T value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;
typedef value_type& reference;
typedef const value_type& const_reference;
typedef std::size_t size_type;
typedef std::ptrdiff_t difference_type;

template<typename U>
struct rebind
{
typedef SVMAllocator<U, SVMTrait> other;
};

template<typename U, typename V>
friend class SVMAllocator;

SVMAllocator() :
context_(Context::getDefault())
{
}

explicit SVMAllocator(cl::Context context) :
context_(context)
{
}


SVMAllocator(const SVMAllocator &other) :
context_(other.context_)
{
}

template<typename U>
SVMAllocator(const SVMAllocator<U, SVMTrait> &other) :
context_(other.context_)
{
}

~SVMAllocator()
{
}

pointer address(reference r) CL_HPP_NOEXCEPT_
{
return std::addressof(r);
}

const_pointer address(const_reference r) CL_HPP_NOEXCEPT_
{
return std::addressof(r);
}


pointer allocate(
size_type size,
typename cl::SVMAllocator<void, SVMTrait>::const_pointer = 0)
{
void* voidPointer =
clSVMAlloc(
context_(),
SVMTrait::getSVMMemFlags(),
size*sizeof(T),
0);
pointer retValue = reinterpret_cast<pointer>(
voidPointer);
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
if (!retValue) {
std::bad_alloc excep;
throw excep;
}
#endif 

if (!(SVMTrait::getSVMMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
cl_int err = enqueueMapSVM(retValue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, size*sizeof(T));
if (err != CL_SUCCESS) {
std::bad_alloc excep;
throw excep;
}
}

return retValue;
}

void deallocate(pointer p, size_type)
{
clSVMFree(context_(), p);
}


size_type max_size() const CL_HPP_NOEXCEPT_
{
size_type maxSize = std::numeric_limits<size_type>::max() / sizeof(T);

for (const Device &d : context_.getInfo<CL_CONTEXT_DEVICES>()) {
maxSize = std::min(
maxSize, 
static_cast<size_type>(d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()));
}

return maxSize;
}

template< class U, class... Args >
void construct(U* p, Args&&... args)
{
new(p)T(args...);
}

template< class U >
void destroy(U* p)
{
p->~U();
}


inline bool operator==(SVMAllocator const& rhs)
{
return (context_==rhs.context_);
}

inline bool operator!=(SVMAllocator const& a)
{
return !operator==(a);
}
}; 


template<class SVMTrait>
class SVMAllocator<void, SVMTrait> {
public:
typedef void value_type;
typedef value_type* pointer;
typedef const value_type* const_pointer;

template<typename U>
struct rebind
{
typedef SVMAllocator<U, SVMTrait> other;
};

template<typename U, typename V>
friend class SVMAllocator;
};

#if !defined(CL_HPP_NO_STD_UNIQUE_PTR)
namespace detail
{
template<class Alloc>
class Deleter {
private:
Alloc alloc_;
size_type copies_;

public:
typedef typename std::allocator_traits<Alloc>::pointer pointer;

Deleter(const Alloc &alloc, size_type copies) : alloc_{ alloc }, copies_{ copies }
{
}

void operator()(pointer ptr) const {
Alloc tmpAlloc{ alloc_ };
std::allocator_traits<Alloc>::destroy(tmpAlloc, std::addressof(*ptr));
std::allocator_traits<Alloc>::deallocate(tmpAlloc, ptr, copies_);
}
};
} 


template <class T, class Alloc, class... Args>
cl::pointer<T, detail::Deleter<Alloc>> allocate_pointer(const Alloc &alloc_, Args&&... args)
{
Alloc alloc(alloc_);
static const size_type copies = 1;


T* tmp = std::allocator_traits<Alloc>::allocate(alloc, copies);
if (!tmp) {
std::bad_alloc excep;
throw excep;
}
try {
std::allocator_traits<Alloc>::construct(
alloc,
std::addressof(*tmp),
std::forward<Args>(args)...);

return cl::pointer<T, detail::Deleter<Alloc>>(tmp, detail::Deleter<Alloc>{alloc, copies});
}
catch (std::bad_alloc&)
{
std::allocator_traits<Alloc>::deallocate(alloc, tmp, copies);
throw;
}
}

template< class T, class SVMTrait, class... Args >
cl::pointer<T, detail::Deleter<SVMAllocator<T, SVMTrait>>> allocate_svm(Args... args)
{
SVMAllocator<T, SVMTrait> alloc;
return cl::allocate_pointer<T>(alloc, args...);
}

template< class T, class SVMTrait, class... Args >
cl::pointer<T, detail::Deleter<SVMAllocator<T, SVMTrait>>> allocate_svm(const cl::Context &c, Args... args)
{
SVMAllocator<T, SVMTrait> alloc(c);
return cl::allocate_pointer<T>(alloc, args...);
}
#endif 


template < class T >
using coarse_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitCoarse<>>>;


template < class T >
using fine_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitFine<>>>;


template < class T >
using atomic_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitAtomic<>>>;

#endif 



class Buffer : public Memory
{
public:


Buffer(
const Context& context,
cl_mem_flags flags,
size_type size,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;
object_ = ::clCreateBuffer(context(), flags, size, host_ptr, &error);

detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}


Buffer(
cl_mem_flags flags,
size_type size,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;

Context context = Context::getDefault(err);

object_ = ::clCreateBuffer(context(), flags, size, host_ptr, &error);

detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}


template< typename IteratorType >
Buffer(
IteratorType startIterator,
IteratorType endIterator,
bool readOnly,
bool useHostPtr = false,
cl_int* err = NULL)
{
typedef typename std::iterator_traits<IteratorType>::value_type DataType;
cl_int error;

cl_mem_flags flags = 0;
if( readOnly ) {
flags |= CL_MEM_READ_ONLY;
}
else {
flags |= CL_MEM_READ_WRITE;
}
if( useHostPtr ) {
flags |= CL_MEM_USE_HOST_PTR;
}

size_type size = sizeof(DataType)*(endIterator - startIterator);

Context context = Context::getDefault(err);

if( useHostPtr ) {
object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
} else {
object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
}

detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}

if( !useHostPtr ) {
error = cl::copy(startIterator, endIterator, *this);
detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}
}


template< typename IteratorType >
Buffer(const Context &context, IteratorType startIterator, IteratorType endIterator,
bool readOnly, bool useHostPtr = false, cl_int* err = NULL);


template< typename IteratorType >
Buffer(const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator,
bool readOnly, bool useHostPtr = false, cl_int* err = NULL);

Buffer() : Memory() { }


explicit Buffer(const cl_mem& buffer, bool retainObject = false) :
Memory(buffer, retainObject) { }


Buffer& operator = (const cl_mem& rhs)
{
Memory::operator=(rhs);
return *this;
}


Buffer(const Buffer& buf) : Memory(buf) {}


Buffer& operator = (const Buffer &buf)
{
Memory::operator=(buf);
return *this;
}


Buffer(Buffer&& buf) CL_HPP_NOEXCEPT_ : Memory(std::move(buf)) {}


Buffer& operator = (Buffer &&buf)
{
Memory::operator=(std::move(buf));
return *this;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 110

Buffer createSubBuffer(
cl_mem_flags flags,
cl_buffer_create_type buffer_create_type,
const void * buffer_create_info,
cl_int * err = NULL)
{
Buffer result;
cl_int error;
result.object_ = ::clCreateSubBuffer(
object_, 
flags, 
buffer_create_type, 
buffer_create_info, 
&error);

detail::errHandler(error, __CREATE_SUBBUFFER_ERR);
if (err != NULL) {
*err = error;
}

return result;
}
#endif 
};

#if defined (CL_HPP_USE_DX_INTEROP)

class BufferD3D10 : public Buffer
{
public:



BufferD3D10(
const Context& context,
cl_mem_flags flags,
ID3D10Buffer* bufobj,
cl_int * err = NULL) : pfn_clCreateFromD3D10BufferKHR(nullptr)
{
typedef CL_API_ENTRY cl_mem (CL_API_CALL *PFN_clCreateFromD3D10BufferKHR)(
cl_context context, cl_mem_flags flags, ID3D10Buffer*  buffer,
cl_int* errcode_ret);
PFN_clCreateFromD3D10BufferKHR pfn_clCreateFromD3D10BufferKHR;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
vector<cl_context_properties> props = context.getInfo<CL_CONTEXT_PROPERTIES>();
cl_platform platform = -1;
for( int i = 0; i < props.size(); ++i ) {
if( props[i] == CL_CONTEXT_PLATFORM ) {
platform = props[i+1];
}
}
CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCreateFromD3D10BufferKHR);
#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateFromD3D10BufferKHR);
#endif

cl_int error;
object_ = pfn_clCreateFromD3D10BufferKHR(
context(),
flags,
bufobj,
&error);

detail::errHandler(error, __CREATE_GL_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}

BufferD3D10() : Buffer() { }


explicit BufferD3D10(const cl_mem& buffer, bool retainObject = false) : 
Buffer(buffer, retainObject) { }


BufferD3D10& operator = (const cl_mem& rhs)
{
Buffer::operator=(rhs);
return *this;
}


BufferD3D10(const BufferD3D10& buf) : 
Buffer(buf) {}


BufferD3D10& operator = (const BufferD3D10 &buf)
{
Buffer::operator=(buf);
return *this;
}


BufferD3D10(BufferD3D10&& buf) CL_HPP_NOEXCEPT_ : Buffer(std::move(buf)) {}


BufferD3D10& operator = (BufferD3D10 &&buf)
{
Buffer::operator=(std::move(buf));
return *this;
}
};
#endif


class BufferGL : public Buffer
{
public:

BufferGL(
const Context& context,
cl_mem_flags flags,
cl_GLuint bufobj,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateFromGLBuffer(
context(),
flags,
bufobj,
&error);

detail::errHandler(error, __CREATE_GL_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}

BufferGL() : Buffer() { }


explicit BufferGL(const cl_mem& buffer, bool retainObject = false) :
Buffer(buffer, retainObject) { }


BufferGL& operator = (const cl_mem& rhs)
{
Buffer::operator=(rhs);
return *this;
}


BufferGL(const BufferGL& buf) : Buffer(buf) {}


BufferGL& operator = (const BufferGL &buf)
{
Buffer::operator=(buf);
return *this;
}


BufferGL(BufferGL&& buf) CL_HPP_NOEXCEPT_ : Buffer(std::move(buf)) {}


BufferGL& operator = (BufferGL &&buf)
{
Buffer::operator=(std::move(buf));
return *this;
}

cl_int getObjectInfo(
cl_gl_object_type *type,
cl_GLuint * gl_object_name)
{
return detail::errHandler(
::clGetGLObjectInfo(object_,type,gl_object_name),
__GET_GL_OBJECT_INFO_ERR);
}
};


class BufferRenderGL : public Buffer
{
public:

BufferRenderGL(
const Context& context,
cl_mem_flags flags,
cl_GLuint bufobj,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateFromGLRenderbuffer(
context(),
flags,
bufobj,
&error);

detail::errHandler(error, __CREATE_GL_RENDER_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}

BufferRenderGL() : Buffer() { }


explicit BufferRenderGL(const cl_mem& buffer, bool retainObject = false) :
Buffer(buffer, retainObject) { }


BufferRenderGL& operator = (const cl_mem& rhs)
{
Buffer::operator=(rhs);
return *this;
}


BufferRenderGL(const BufferRenderGL& buf) : Buffer(buf) {}


BufferRenderGL& operator = (const BufferRenderGL &buf)
{
Buffer::operator=(buf);
return *this;
}


BufferRenderGL(BufferRenderGL&& buf) CL_HPP_NOEXCEPT_ : Buffer(std::move(buf)) {}


BufferRenderGL& operator = (BufferRenderGL &&buf)
{
Buffer::operator=(std::move(buf));
return *this;
}

cl_int getObjectInfo(
cl_gl_object_type *type,
cl_GLuint * gl_object_name)
{
return detail::errHandler(
::clGetGLObjectInfo(object_,type,gl_object_name),
__GET_GL_OBJECT_INFO_ERR);
}
};


class Image : public Memory
{
protected:
Image() : Memory() { }


explicit Image(const cl_mem& image, bool retainObject = false) :
Memory(image, retainObject) { }


Image& operator = (const cl_mem& rhs)
{
Memory::operator=(rhs);
return *this;
}


Image(const Image& img) : Memory(img) {}


Image& operator = (const Image &img)
{
Memory::operator=(img);
return *this;
}


Image(Image&& img) CL_HPP_NOEXCEPT_ : Memory(std::move(img)) {}


Image& operator = (Image &&img)
{
Memory::operator=(std::move(img));
return *this;
}


public:
template <typename T>
cl_int getImageInfo(cl_image_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetImageInfo, object_, name, param),
__GET_IMAGE_INFO_ERR);
}

template <cl_image_info name> typename
detail::param_traits<detail::cl_image_info, name>::param_type
getImageInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_image_info, name>::param_type param;
cl_int result = getImageInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

class Image1D : public Image
{
public:

Image1D(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type width,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE1D;
desc.image_width = width;

object_ = ::clCreateImage(
context(), 
flags, 
&format, 
&desc, 
host_ptr, 
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}

Image1D() { }


explicit Image1D(const cl_mem& image1D, bool retainObject = false) :
Image(image1D, retainObject) { }


Image1D& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image1D(const Image1D& img) : Image(img) {}


Image1D& operator = (const Image1D &img)
{
Image::operator=(img);
return *this;
}


Image1D(Image1D&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image1D& operator = (Image1D &&img)
{
Image::operator=(std::move(img));
return *this;
}

};


class Image1DBuffer : public Image
{
public:
Image1DBuffer(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type width,
const Buffer &buffer,
cl_int* err = NULL)
{
cl_int error;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
desc.image_width = width;
desc.buffer = buffer();

object_ = ::clCreateImage(
context(), 
flags, 
&format, 
&desc, 
NULL, 
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}

Image1DBuffer() { }


explicit Image1DBuffer(const cl_mem& image1D, bool retainObject = false) :
Image(image1D, retainObject) { }

Image1DBuffer& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image1DBuffer(const Image1DBuffer& img) : Image(img) {}


Image1DBuffer& operator = (const Image1DBuffer &img)
{
Image::operator=(img);
return *this;
}


Image1DBuffer(Image1DBuffer&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image1DBuffer& operator = (Image1DBuffer &&img)
{
Image::operator=(std::move(img));
return *this;
}

};


class Image1DArray : public Image
{
public:
Image1DArray(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type arraySize,
size_type width,
size_type rowPitch,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
desc.image_width = width;
desc.image_array_size = arraySize;
desc.image_row_pitch = rowPitch;

object_ = ::clCreateImage(
context(), 
flags, 
&format, 
&desc, 
host_ptr, 
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}

Image1DArray() { }


explicit Image1DArray(const cl_mem& imageArray, bool retainObject = false) :
Image(imageArray, retainObject) { }


Image1DArray& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image1DArray(const Image1DArray& img) : Image(img) {}


Image1DArray& operator = (const Image1DArray &img)
{
Image::operator=(img);
return *this;
}


Image1DArray(Image1DArray&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image1DArray& operator = (Image1DArray &&img)
{
Image::operator=(std::move(img));
return *this;
}

};
#endif 



class Image2D : public Image
{
public:

Image2D(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type width,
size_type height,
size_type row_pitch = 0,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;
bool useCreateImage;

#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
{
cl_uint version = detail::getContextPlatformVersion(context());
useCreateImage = (version >= 0x10002); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
useCreateImage = true;
#else
useCreateImage = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
if (useCreateImage)
{
cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE2D;
desc.image_width = width;
desc.image_height = height;
desc.image_row_pitch = row_pitch;

object_ = ::clCreateImage(
context(),
flags,
&format,
&desc,
host_ptr,
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
if (!useCreateImage)
{
object_ = ::clCreateImage2D(
context(), flags,&format, width, height, row_pitch, host_ptr, &error);

detail::errHandler(error, __CREATE_IMAGE2D_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)

Image2D(
const Context& context,
ImageFormat format,
const Buffer &sourceBuffer,
size_type width,
size_type height,
size_type row_pitch = 0,
cl_int* err = nullptr)
{
cl_int error;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE2D;
desc.image_width = width;
desc.image_height = height;
desc.image_row_pitch = row_pitch;
desc.buffer = sourceBuffer();

object_ = ::clCreateImage(
context(),
0, 
&format,
&desc,
nullptr,
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != nullptr) {
*err = error;
}
}
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

Image2D(
const Context& context,
cl_channel_order order,
const Image &sourceImage,
cl_int* err = nullptr)
{
cl_int error;

size_type sourceWidth = 
sourceImage.getImageInfo<CL_IMAGE_WIDTH>();
size_type sourceHeight = 
sourceImage.getImageInfo<CL_IMAGE_HEIGHT>();
size_type sourceRowPitch =
sourceImage.getImageInfo<CL_IMAGE_ROW_PITCH>();
cl_uint sourceNumMIPLevels =
sourceImage.getImageInfo<CL_IMAGE_NUM_MIP_LEVELS>();
cl_uint sourceNumSamples =
sourceImage.getImageInfo<CL_IMAGE_NUM_SAMPLES>();
cl_image_format sourceFormat =
sourceImage.getImageInfo<CL_IMAGE_FORMAT>();

sourceFormat.image_channel_order = order;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE2D;
desc.image_width = sourceWidth;
desc.image_height = sourceHeight;
desc.image_row_pitch = sourceRowPitch;
desc.num_mip_levels = sourceNumMIPLevels;
desc.num_samples = sourceNumSamples;
desc.buffer = sourceImage();

object_ = ::clCreateImage(
context(),
0, 
&sourceFormat,
&desc,
nullptr,
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != nullptr) {
*err = error;
}
}
#endif 

Image2D() { }


explicit Image2D(const cl_mem& image2D, bool retainObject = false) :
Image(image2D, retainObject) { }


Image2D& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image2D(const Image2D& img) : Image(img) {}


Image2D& operator = (const Image2D &img)
{
Image::operator=(img);
return *this;
}


Image2D(Image2D&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image2D& operator = (Image2D &&img)
{
Image::operator=(std::move(img));
return *this;
}

};


#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

class CL_API_PREFIX__VERSION_1_1_DEPRECATED Image2DGL : public Image2D 
{
public:

Image2DGL(
const Context& context,
cl_mem_flags flags,
cl_GLenum target,
cl_GLint  miplevel,
cl_GLuint texobj,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateFromGLTexture2D(
context(),
flags,
target,
miplevel,
texobj,
&error);

detail::errHandler(error, __CREATE_GL_TEXTURE_2D_ERR);
if (err != NULL) {
*err = error;
}

}

Image2DGL() : Image2D() { }


explicit Image2DGL(const cl_mem& image, bool retainObject = false) : 
Image2D(image, retainObject) { }


Image2DGL& operator = (const cl_mem& rhs)
{
Image2D::operator=(rhs);
return *this;
}


Image2DGL(const Image2DGL& img) : Image2D(img) {}


Image2DGL& operator = (const Image2DGL &img)
{
Image2D::operator=(img);
return *this;
}


Image2DGL(Image2DGL&& img) CL_HPP_NOEXCEPT_ : Image2D(std::move(img)) {}


Image2DGL& operator = (Image2DGL &&img)
{
Image2D::operator=(std::move(img));
return *this;
}

} CL_API_SUFFIX__VERSION_1_1_DEPRECATED;
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

class Image2DArray : public Image
{
public:
Image2DArray(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type arraySize,
size_type width,
size_type height,
size_type rowPitch,
size_type slicePitch,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;

cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
desc.image_width = width;
desc.image_height = height;
desc.image_array_size = arraySize;
desc.image_row_pitch = rowPitch;
desc.image_slice_pitch = slicePitch;

object_ = ::clCreateImage(
context(), 
flags, 
&format, 
&desc, 
host_ptr, 
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}

Image2DArray() { }


explicit Image2DArray(const cl_mem& imageArray, bool retainObject = false) : Image(imageArray, retainObject) { }

Image2DArray& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image2DArray(const Image2DArray& img) : Image(img) {}


Image2DArray& operator = (const Image2DArray &img)
{
Image::operator=(img);
return *this;
}


Image2DArray(Image2DArray&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image2DArray& operator = (Image2DArray &&img)
{
Image::operator=(std::move(img));
return *this;
}
};
#endif 


class Image3D : public Image
{
public:

Image3D(
const Context& context,
cl_mem_flags flags,
ImageFormat format,
size_type width,
size_type height,
size_type depth,
size_type row_pitch = 0,
size_type slice_pitch = 0,
void* host_ptr = NULL,
cl_int* err = NULL)
{
cl_int error;
bool useCreateImage;

#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
{
cl_uint version = detail::getContextPlatformVersion(context());
useCreateImage = (version >= 0x10002); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
useCreateImage = true;
#else
useCreateImage = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
if (useCreateImage)
{
cl_image_desc desc = {0};
desc.image_type = CL_MEM_OBJECT_IMAGE3D;
desc.image_width = width;
desc.image_height = height;
desc.image_depth = depth;
desc.image_row_pitch = row_pitch;
desc.image_slice_pitch = slice_pitch;

object_ = ::clCreateImage(
context(), 
flags, 
&format, 
&desc, 
host_ptr, 
&error);

detail::errHandler(error, __CREATE_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif  
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
if (!useCreateImage)
{
object_ = ::clCreateImage3D(
context(), flags, &format, width, height, depth, row_pitch,
slice_pitch, host_ptr, &error);

detail::errHandler(error, __CREATE_IMAGE3D_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}

Image3D() : Image() { }


explicit Image3D(const cl_mem& image3D, bool retainObject = false) : 
Image(image3D, retainObject) { }


Image3D& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


Image3D(const Image3D& img) : Image(img) {}


Image3D& operator = (const Image3D &img)
{
Image::operator=(img);
return *this;
}


Image3D(Image3D&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


Image3D& operator = (Image3D &&img)
{
Image::operator=(std::move(img));
return *this;
}
};

#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

class Image3DGL : public Image3D
{
public:

Image3DGL(
const Context& context,
cl_mem_flags flags,
cl_GLenum target,
cl_GLint  miplevel,
cl_GLuint texobj,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateFromGLTexture3D(
context(),
flags,
target,
miplevel,
texobj,
&error);

detail::errHandler(error, __CREATE_GL_TEXTURE_3D_ERR);
if (err != NULL) {
*err = error;
}
}

Image3DGL() : Image3D() { }


explicit Image3DGL(const cl_mem& image, bool retainObject = false) : 
Image3D(image, retainObject) { }


Image3DGL& operator = (const cl_mem& rhs)
{
Image3D::operator=(rhs);
return *this;
}


Image3DGL(const Image3DGL& img) : Image3D(img) {}


Image3DGL& operator = (const Image3DGL &img)
{
Image3D::operator=(img);
return *this;
}


Image3DGL(Image3DGL&& img) CL_HPP_NOEXCEPT_ : Image3D(std::move(img)) {}


Image3DGL& operator = (Image3DGL &&img)
{
Image3D::operator=(std::move(img));
return *this;
}
};
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

class ImageGL : public Image
{
public:
ImageGL(
const Context& context,
cl_mem_flags flags,
cl_GLenum target,
cl_GLint  miplevel,
cl_GLuint texobj,
cl_int * err = NULL)
{
cl_int error;
object_ = ::clCreateFromGLTexture(
context(), 
flags, 
target,
miplevel,
texobj,
&error);

detail::errHandler(error, __CREATE_GL_TEXTURE_ERR);
if (err != NULL) {
*err = error;
}
}

ImageGL() : Image() { }


explicit ImageGL(const cl_mem& image, bool retainObject = false) : 
Image(image, retainObject) { }

ImageGL& operator = (const cl_mem& rhs)
{
Image::operator=(rhs);
return *this;
}


ImageGL(const ImageGL& img) : Image(img) {}


ImageGL& operator = (const ImageGL &img)
{
Image::operator=(img);
return *this;
}


ImageGL(ImageGL&& img) CL_HPP_NOEXCEPT_ : Image(std::move(img)) {}


ImageGL& operator = (ImageGL &&img)
{
Image::operator=(std::move(img));
return *this;
}
};
#endif 



#if CL_HPP_TARGET_OPENCL_VERSION >= 200

class Pipe : public Memory
{
public:


Pipe(
const Context& context,
cl_uint packet_size,
cl_uint max_packets,
cl_int* err = NULL)
{
cl_int error;

cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
object_ = ::clCreatePipe(context(), flags, packet_size, max_packets, nullptr, &error);

detail::errHandler(error, __CREATE_PIPE_ERR);
if (err != NULL) {
*err = error;
}
}


Pipe(
cl_uint packet_size,
cl_uint max_packets,
cl_int* err = NULL)
{
cl_int error;

Context context = Context::getDefault(err);

cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
object_ = ::clCreatePipe(context(), flags, packet_size, max_packets, nullptr, &error);

detail::errHandler(error, __CREATE_PIPE_ERR);
if (err != NULL) {
*err = error;
}
}

Pipe() : Memory() { }


explicit Pipe(const cl_mem& pipe, bool retainObject = false) :
Memory(pipe, retainObject) { }


Pipe& operator = (const cl_mem& rhs)
{
Memory::operator=(rhs);
return *this;
}


Pipe(const Pipe& pipe) : Memory(pipe) {}


Pipe& operator = (const Pipe &pipe)
{
Memory::operator=(pipe);
return *this;
}


Pipe(Pipe&& pipe) CL_HPP_NOEXCEPT_ : Memory(std::move(pipe)) {}


Pipe& operator = (Pipe &&pipe)
{
Memory::operator=(std::move(pipe));
return *this;
}

template <typename T>
cl_int getInfo(cl_pipe_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetPipeInfo, object_, name, param),
__GET_PIPE_INFO_ERR);
}

template <cl_pipe_info name> typename
detail::param_traits<detail::cl_pipe_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_pipe_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}
}; 
#endif 



class Sampler : public detail::Wrapper<cl_sampler>
{
public:
Sampler() { }


Sampler(
const Context& context,
cl_bool normalized_coords,
cl_addressing_mode addressing_mode,
cl_filter_mode filter_mode,
cl_int* err = NULL)
{
cl_int error;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
cl_sampler_properties sampler_properties[] = {
CL_SAMPLER_NORMALIZED_COORDS, normalized_coords,
CL_SAMPLER_ADDRESSING_MODE, addressing_mode,
CL_SAMPLER_FILTER_MODE, filter_mode,
0 };
object_ = ::clCreateSamplerWithProperties(
context(),
sampler_properties,
&error);

detail::errHandler(error, __CREATE_SAMPLER_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
#else
object_ = ::clCreateSampler(
context(),
normalized_coords,
addressing_mode,
filter_mode,
&error);

detail::errHandler(error, __CREATE_SAMPLER_ERR);
if (err != NULL) {
*err = error;
}
#endif        
}


explicit Sampler(const cl_sampler& sampler, bool retainObject = false) : 
detail::Wrapper<cl_type>(sampler, retainObject) { }


Sampler& operator = (const cl_sampler& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


Sampler(const Sampler& sam) : detail::Wrapper<cl_type>(sam) {}


Sampler& operator = (const Sampler &sam)
{
detail::Wrapper<cl_type>::operator=(sam);
return *this;
}


Sampler(Sampler&& sam) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(sam)) {}


Sampler& operator = (Sampler &&sam)
{
detail::Wrapper<cl_type>::operator=(std::move(sam));
return *this;
}

template <typename T>
cl_int getInfo(cl_sampler_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetSamplerInfo, object_, name, param),
__GET_SAMPLER_INFO_ERR);
}

template <cl_sampler_info name> typename
detail::param_traits<detail::cl_sampler_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_sampler_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}
};

class Program;
class CommandQueue;
class DeviceCommandQueue;
class Kernel;

class NDRange
{
private:
size_type sizes_[3];
cl_uint dimensions_;

public:
NDRange()
: dimensions_(0)
{
sizes_[0] = 0;
sizes_[1] = 0;
sizes_[2] = 0;
}

NDRange(size_type size0)
: dimensions_(1)
{
sizes_[0] = size0;
sizes_[1] = 1;
sizes_[2] = 1;
}

NDRange(size_type size0, size_type size1)
: dimensions_(2)
{
sizes_[0] = size0;
sizes_[1] = size1;
sizes_[2] = 1;
}

NDRange(size_type size0, size_type size1, size_type size2)
: dimensions_(3)
{
sizes_[0] = size0;
sizes_[1] = size1;
sizes_[2] = size2;
}


operator const size_type*() const { 
return sizes_; 
}

size_type dimensions() const 
{ 
return dimensions_; 
}

size_type size() const
{
return dimensions_*sizeof(size_type);
}

size_type* get()
{
return sizes_;
}

const size_type* get() const
{
return sizes_;
}
};

static const NDRange NullRange;

struct LocalSpaceArg
{
size_type size_;
};

namespace detail {

template <typename T, class Enable = void>
struct KernelArgumentHandler;

template <typename T>
struct KernelArgumentHandler<T, typename std::enable_if<!std::is_base_of<cl::Memory, T>::value>::type>
{
static size_type size(const T&) { return sizeof(T); }
static const T* ptr(const T& value) { return &value; }
};

template <typename T>
struct KernelArgumentHandler<T, typename std::enable_if<std::is_base_of<cl::Memory, T>::value>::type>
{
static size_type size(const T&) { return sizeof(cl_mem); }
static const cl_mem* ptr(const T& value) { return &(value()); }
};


template <>
struct KernelArgumentHandler<LocalSpaceArg, void>
{
static size_type size(const LocalSpaceArg& value) { return value.size_; }
static const void* ptr(const LocalSpaceArg&) { return NULL; }
};

} 


inline LocalSpaceArg
Local(size_type size)
{
LocalSpaceArg ret = { size };
return ret;
}


class Kernel : public detail::Wrapper<cl_kernel>
{
public:
inline Kernel(const Program& program, const char* name, cl_int* err = NULL);

Kernel() { }


explicit Kernel(const cl_kernel& kernel, bool retainObject = false) : 
detail::Wrapper<cl_type>(kernel, retainObject) { }


Kernel& operator = (const cl_kernel& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


Kernel(const Kernel& kernel) : detail::Wrapper<cl_type>(kernel) {}


Kernel& operator = (const Kernel &kernel)
{
detail::Wrapper<cl_type>::operator=(kernel);
return *this;
}


Kernel(Kernel&& kernel) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(kernel)) {}


Kernel& operator = (Kernel &&kernel)
{
detail::Wrapper<cl_type>::operator=(std::move(kernel));
return *this;
}

template <typename T>
cl_int getInfo(cl_kernel_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetKernelInfo, object_, name, param),
__GET_KERNEL_INFO_ERR);
}

template <cl_kernel_info name> typename
detail::param_traits<detail::cl_kernel_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_kernel_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
template <typename T>
cl_int getArgInfo(cl_uint argIndex, cl_kernel_arg_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetKernelArgInfo, object_, argIndex, name, param),
__GET_KERNEL_ARG_INFO_ERR);
}

template <cl_kernel_arg_info name> typename
detail::param_traits<detail::cl_kernel_arg_info, name>::param_type
getArgInfo(cl_uint argIndex, cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_kernel_arg_info, name>::param_type param;
cl_int result = getArgInfo(argIndex, name, &param);
if (err != NULL) {
*err = result;
}
return param;
}
#endif 

template <typename T>
cl_int getWorkGroupInfo(
const Device& device, cl_kernel_work_group_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(
&::clGetKernelWorkGroupInfo, object_, device(), name, param),
__GET_KERNEL_WORK_GROUP_INFO_ERR);
}

template <cl_kernel_work_group_info name> typename
detail::param_traits<detail::cl_kernel_work_group_info, name>::param_type
getWorkGroupInfo(const Device& device, cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_kernel_work_group_info, name>::param_type param;
cl_int result = getWorkGroupInfo(device, name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

#if (CL_HPP_TARGET_OPENCL_VERSION >= 200 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)) || CL_HPP_TARGET_OPENCL_VERSION >= 210
cl_int getSubGroupInfo(const cl::Device &dev, cl_kernel_sub_group_info name, const cl::NDRange &range, size_type* param) const
{
#if CL_HPP_TARGET_OPENCL_VERSION >= 210

return detail::errHandler(
clGetKernelSubGroupInfo(object_, dev(), name, range.size(), range.get(), sizeof(size_type), param, nullptr),
__GET_KERNEL_SUB_GROUP_INFO_ERR);

#else 

typedef clGetKernelSubGroupInfoKHR_fn PFN_clGetKernelSubGroupInfoKHR;
static PFN_clGetKernelSubGroupInfoKHR pfn_clGetKernelSubGroupInfoKHR = NULL;
CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetKernelSubGroupInfoKHR);

return detail::errHandler(
pfn_clGetKernelSubGroupInfoKHR(object_, dev(), name, range.size(), range.get(), sizeof(size_type), param, nullptr),
__GET_KERNEL_SUB_GROUP_INFO_ERR);

#endif 
}

template <cl_kernel_sub_group_info name>
size_type getSubGroupInfo(const cl::Device &dev, const cl::NDRange &range, cl_int* err = NULL) const
{
size_type param;
cl_int result = getSubGroupInfo(dev, name, range, &param);
if (err != NULL) {
*err = result;
}
return param;
}
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T, class D>
cl_int setArg(cl_uint index, const cl::pointer<T, D> &argPtr)
{
return detail::errHandler(
::clSetKernelArgSVMPointer(object_, index, argPtr.get()),
__SET_KERNEL_ARGS_ERR);
}


template<typename T, class Alloc>
cl_int setArg(cl_uint index, const cl::vector<T, Alloc> &argPtr)
{
return detail::errHandler(
::clSetKernelArgSVMPointer(object_, index, argPtr.data()),
__SET_KERNEL_ARGS_ERR);
}


template<typename T>
typename std::enable_if<std::is_pointer<T>::value, cl_int>::type
setArg(cl_uint index, const T argPtr)
{
return detail::errHandler(
::clSetKernelArgSVMPointer(object_, index, argPtr),
__SET_KERNEL_ARGS_ERR);
}
#endif 


template <typename T>
typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type
setArg(cl_uint index, const T &value)
{
return detail::errHandler(
::clSetKernelArg(
object_,
index,
detail::KernelArgumentHandler<T>::size(value),
detail::KernelArgumentHandler<T>::ptr(value)),
__SET_KERNEL_ARGS_ERR);
}

cl_int setArg(cl_uint index, size_type size, const void* argPtr)
{
return detail::errHandler(
::clSetKernelArg(object_, index, size, argPtr),
__SET_KERNEL_ARGS_ERR);
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

cl_int setSVMPointers(const vector<void*> &pointerList)
{
return detail::errHandler(
::clSetKernelExecInfo(
object_,
CL_KERNEL_EXEC_INFO_SVM_PTRS,
sizeof(void*)*pointerList.size(),
pointerList.data()));
}


template<int ArrayLength>
cl_int setSVMPointers(const std::array<void*, ArrayLength> &pointerList)
{
return detail::errHandler(
::clSetKernelExecInfo(
object_,
CL_KERNEL_EXEC_INFO_SVM_PTRS,
sizeof(void*)*pointerList.size(),
pointerList.data()));
}


cl_int enableFineGrainedSystemSVM(bool svmEnabled)
{
cl_bool svmEnabled_ = svmEnabled ? CL_TRUE : CL_FALSE;
return detail::errHandler(
::clSetKernelExecInfo(
object_,
CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
sizeof(cl_bool),
&svmEnabled_
)
);
}

template<int index, int ArrayLength, class D, typename T0, typename T1, typename... Ts>
void setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, const pointer<T0, D> &t0, const pointer<T1, D> &t1, Ts & ... ts)
{
pointerList[index] = static_cast<void*>(t0.get());
setSVMPointersHelper<index + 1, ArrayLength>(pointerList, t1, ts...);
}

template<int index, int ArrayLength, typename T0, typename T1, typename... Ts>
typename std::enable_if<std::is_pointer<T0>::value, void>::type
setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, T0 t0, T1 t1, Ts... ts)
{
pointerList[index] = static_cast<void*>(t0);
setSVMPointersHelper<index + 1, ArrayLength>(pointerList, t1, ts...);
}

template<int index, int ArrayLength, typename T0, class D>
void setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, const pointer<T0, D> &t0)
{
pointerList[index] = static_cast<void*>(t0.get());
}


template<int index, int ArrayLength, typename T0>
typename std::enable_if<std::is_pointer<T0>::value, void>::type
setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, T0 t0)
{
pointerList[index] = static_cast<void*>(t0);
}

template<typename T0, typename... Ts>
cl_int setSVMPointers(const T0 &t0, Ts & ... ts)
{
std::array<void*, 1 + sizeof...(Ts)> pointerList;

setSVMPointersHelper<0, 1 + sizeof...(Ts)>(pointerList, t0, ts...);
return detail::errHandler(
::clSetKernelExecInfo(
object_,
CL_KERNEL_EXEC_INFO_SVM_PTRS,
sizeof(void*)*(1 + sizeof...(Ts)),
pointerList.data()));
}

template<typename T>
cl_int setExecInfo(cl_kernel_exec_info param_name, const T& val)
{
return detail::errHandler(
::clSetKernelExecInfo(
object_,
param_name,
sizeof(T),
&val));
}

template<cl_kernel_exec_info name>
cl_int setExecInfo(typename detail::param_traits<detail::cl_kernel_exec_info, name>::param_type& val)
{
return setExecInfo(name, val);
}
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 210

Kernel clone()
{
cl_int error;
Kernel retValue(clCloneKernel(this->get(), &error));

detail::errHandler(error, __CLONE_KERNEL_ERR);
return retValue;
}
#endif 
};


class Program : public detail::Wrapper<cl_program>
{
public:
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
typedef vector<vector<unsigned char>> Binaries;
typedef vector<string> Sources;
#else 
typedef vector<std::pair<const void*, size_type> > Binaries;
typedef vector<std::pair<const char*, size_type> > Sources;
#endif 

Program(
const string& source,
bool build = false,
cl_int* err = NULL)
{
cl_int error;

const char * strings = source.c_str();
const size_type length  = source.size();

Context context = Context::getDefault(err);

object_ = ::clCreateProgramWithSource(
context(), (cl_uint)1, &strings, &length, &error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);

if (error == CL_SUCCESS && build) {

error = ::clBuildProgram(
object_,
0,
NULL,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
"-cl-std=CL2.0",
#else
"",
#endif 
NULL,
NULL);

detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

if (err != NULL) {
*err = error;
}
}

Program(
const Context& context,
const string& source,
bool build = false,
cl_int* err = NULL)
{
cl_int error;

const char * strings = source.c_str();
const size_type length  = source.size();

object_ = ::clCreateProgramWithSource(
context(), (cl_uint)1, &strings, &length, &error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);

if (error == CL_SUCCESS && build) {
error = ::clBuildProgram(
object_,
0,
NULL,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
"-cl-std=CL2.0",
#else
"",
#endif 
NULL,
NULL);

detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

if (err != NULL) {
*err = error;
}
}


Program(
const Sources& sources,
cl_int* err = NULL)
{
cl_int error;
Context context = Context::getDefault(err);

const size_type n = (size_type)sources.size();

vector<size_type> lengths(n);
vector<const char*> strings(n);

for (size_type i = 0; i < n; ++i) {
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
strings[i] = sources[(int)i].data();
lengths[i] = sources[(int)i].length();
#else 
strings[i] = sources[(int)i].first;
lengths[i] = sources[(int)i].second;
#endif 
}

object_ = ::clCreateProgramWithSource(
context(), (cl_uint)n, strings.data(), lengths.data(), &error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);
if (err != NULL) {
*err = error;
}
}


Program(
const Context& context,
const Sources& sources,
cl_int* err = NULL)
{
cl_int error;

const size_type n = (size_type)sources.size();

vector<size_type> lengths(n);
vector<const char*> strings(n);

for (size_type i = 0; i < n; ++i) {
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
strings[i] = sources[(int)i].data();
lengths[i] = sources[(int)i].length();
#else 
strings[i] = sources[(int)i].first;
lengths[i] = sources[(int)i].second;
#endif 
}

object_ = ::clCreateProgramWithSource(
context(), (cl_uint)n, strings.data(), lengths.data(), &error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);
if (err != NULL) {
*err = error;
}
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 210 || (CL_HPP_TARGET_OPENCL_VERSION==200 && defined(CL_HPP_USE_IL_KHR))

Program(
const vector<char>& IL,
bool build = false,
cl_int* err = NULL)
{
cl_int error;

Context context = Context::getDefault(err);

#if CL_HPP_TARGET_OPENCL_VERSION >= 210

object_ = ::clCreateProgramWithIL(
context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#else 

typedef clCreateProgramWithILKHR_fn PFN_clCreateProgramWithILKHR;
static PFN_clCreateProgramWithILKHR pfn_clCreateProgramWithILKHR = NULL;
CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateProgramWithILKHR);

object_ = pfn_clCreateProgramWithILKHR(
context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#endif 

detail::errHandler(error, __CREATE_PROGRAM_WITH_IL_ERR);

if (error == CL_SUCCESS && build) {

error = ::clBuildProgram(
object_,
0,
NULL,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
"-cl-std=CL2.0",
#else
"",
#endif 
NULL,
NULL);

detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

if (err != NULL) {
*err = error;
}
}


Program(
const Context& context,
const vector<char>& IL,
bool build = false,
cl_int* err = NULL)
{
cl_int error;

#if CL_HPP_TARGET_OPENCL_VERSION >= 210

object_ = ::clCreateProgramWithIL(
context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#else 

typedef clCreateProgramWithILKHR_fn PFN_clCreateProgramWithILKHR;
static PFN_clCreateProgramWithILKHR pfn_clCreateProgramWithILKHR = NULL;
CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateProgramWithILKHR);

object_ = pfn_clCreateProgramWithILKHR(
context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#endif 

detail::errHandler(error, __CREATE_PROGRAM_WITH_IL_ERR);

if (error == CL_SUCCESS && build) {
error = ::clBuildProgram(
object_,
0,
NULL,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
"-cl-std=CL2.0",
#else
"",
#endif 
NULL,
NULL);

detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

if (err != NULL) {
*err = error;
}
}
#endif 


Program(
const Context& context,
const vector<Device>& devices,
const Binaries& binaries,
vector<cl_int>* binaryStatus = NULL,
cl_int* err = NULL)
{
cl_int error;

const size_type numDevices = devices.size();

if(binaries.size() != numDevices) {
error = CL_INVALID_VALUE;
detail::errHandler(error, __CREATE_PROGRAM_WITH_BINARY_ERR);
if (err != NULL) {
*err = error;
}
return;
}


vector<size_type> lengths(numDevices);
vector<const unsigned char*> images(numDevices);
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
for (size_type i = 0; i < numDevices; ++i) {
images[i] = binaries[i].data();
lengths[i] = binaries[(int)i].size();
}
#else 
for (size_type i = 0; i < numDevices; ++i) {
images[i] = (const unsigned char*)binaries[i].first;
lengths[i] = binaries[(int)i].second;
}
#endif 

vector<cl_device_id> deviceIDs(numDevices);
for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
deviceIDs[deviceIndex] = (devices[deviceIndex])();
}

if(binaryStatus) {
binaryStatus->resize(numDevices);
}

object_ = ::clCreateProgramWithBinary(
context(), (cl_uint) devices.size(),
deviceIDs.data(),
lengths.data(), images.data(), (binaryStatus != NULL && numDevices > 0)
? &binaryStatus->front()
: NULL, &error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_BINARY_ERR);
if (err != NULL) {
*err = error;
}
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 120

Program(
const Context& context,
const vector<Device>& devices,
const string& kernelNames,
cl_int* err = NULL)
{
cl_int error;


size_type numDevices = devices.size();
vector<cl_device_id> deviceIDs(numDevices);
for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
deviceIDs[deviceIndex] = (devices[deviceIndex])();
}

object_ = ::clCreateProgramWithBuiltInKernels(
context(), 
(cl_uint) devices.size(),
deviceIDs.data(),
kernelNames.c_str(), 
&error);

detail::errHandler(error, __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 

Program() { }



explicit Program(const cl_program& program, bool retainObject = false) : 
detail::Wrapper<cl_type>(program, retainObject) { }

Program& operator = (const cl_program& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


Program(const Program& program) : detail::Wrapper<cl_type>(program) {}


Program& operator = (const Program &program)
{
detail::Wrapper<cl_type>::operator=(program);
return *this;
}


Program(Program&& program) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(program)) {}


Program& operator = (Program &&program)
{
detail::Wrapper<cl_type>::operator=(std::move(program));
return *this;
}

cl_int build(
const vector<Device>& devices,
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL) const
{
size_type numDevices = devices.size();
vector<cl_device_id> deviceIDs(numDevices);

for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
deviceIDs[deviceIndex] = (devices[deviceIndex])();
}

cl_int buildError = ::clBuildProgram(
object_,
(cl_uint)
devices.size(),
deviceIDs.data(),
options,
notifyFptr,
data);

return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

cl_int build(
const Device& device,
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL) const
{
cl_device_id deviceID = device();

cl_int buildError = ::clBuildProgram(
object_,
1,
&deviceID,
options,
notifyFptr,
data);

BuildLogType buildLog(0);
buildLog.push_back(std::make_pair(device, getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)));
return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, buildLog);
}

cl_int build(
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL) const
{
cl_int buildError = ::clBuildProgram(
object_,
0,
NULL,
options,
notifyFptr,
data);

return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
cl_int compile(
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL) const
{
cl_int error = ::clCompileProgram(
object_,
0,
NULL,
options,
0,
NULL,
NULL,
notifyFptr,
data);
return detail::buildErrHandler(error, __COMPILE_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
}
#endif 

template <typename T>
cl_int getInfo(cl_program_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(&::clGetProgramInfo, object_, name, param),
__GET_PROGRAM_INFO_ERR);
}

template <cl_program_info name> typename
detail::param_traits<detail::cl_program_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_program_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

template <typename T>
cl_int getBuildInfo(
const Device& device, cl_program_build_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(
&::clGetProgramBuildInfo, object_, device(), name, param),
__GET_PROGRAM_BUILD_INFO_ERR);
}

template <cl_program_build_info name> typename
detail::param_traits<detail::cl_program_build_info, name>::param_type
getBuildInfo(const Device& device, cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_program_build_info, name>::param_type param;
cl_int result = getBuildInfo(device, name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


template <cl_program_build_info name>
vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>>
getBuildInfo(cl_int *err = NULL) const
{
cl_int result = CL_SUCCESS;

auto devs = getInfo<CL_PROGRAM_DEVICES>(&result);
vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>>
devInfo;

if (result != CL_SUCCESS) {
if (err != NULL) {
*err = result;
}
return devInfo;
}

for (const cl::Device &d : devs) {
typename detail::param_traits<
detail::cl_program_build_info, name>::param_type param;
result = getBuildInfo(d, name, &param);
devInfo.push_back(
std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>
(d, param));
if (result != CL_SUCCESS) {
break;
}
}
if (err != NULL) {
*err = result;
}
if (result != CL_SUCCESS) {
devInfo.clear();
}
return devInfo;
}

cl_int createKernels(vector<Kernel>* kernels)
{
cl_uint numKernels;
cl_int err = ::clCreateKernelsInProgram(object_, 0, NULL, &numKernels);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_KERNELS_IN_PROGRAM_ERR);
}

vector<cl_kernel> value(numKernels);

err = ::clCreateKernelsInProgram(
object_, numKernels, value.data(), NULL);
if (err != CL_SUCCESS) {
return detail::errHandler(err, __CREATE_KERNELS_IN_PROGRAM_ERR);
}

if (kernels) {
kernels->resize(value.size());

for (size_type i = 0; i < value.size(); i++) {
(*kernels)[i] = Kernel(value[i], false);
}
}
return CL_SUCCESS;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 220
#if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)

CL_API_PREFIX__VERSION_2_2_DEPRECATED cl_int setReleaseCallback(
void (CL_CALLBACK * pfn_notify)(cl_program program, void * user_data),
void * user_data = NULL) CL_API_SUFFIX__VERSION_2_2_DEPRECATED
{
return detail::errHandler(
::clSetProgramReleaseCallback(
object_,
pfn_notify,
user_data),
__SET_PROGRAM_RELEASE_CALLBACK_ERR);
}
#endif 


template <typename T>
typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type
setSpecializationConstant(cl_uint index, const T &value)
{
return detail::errHandler(
::clSetProgramSpecializationConstant(
object_,
index,
sizeof(value),
&value),
__SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
}


cl_int setSpecializationConstant(cl_uint index, size_type size, const void* value)
{
return detail::errHandler(
::clSetProgramSpecializationConstant(
object_,
index,
size,
value),
__SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
}
#endif 
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
inline Program linkProgram(
Program input1,
Program input2,
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL,
cl_int* err = NULL) 
{
cl_int error_local = CL_SUCCESS;

cl_program programs[2] = { input1(), input2() };

Context ctx = input1.getInfo<CL_PROGRAM_CONTEXT>(&error_local);
if(error_local!=CL_SUCCESS) {
detail::errHandler(error_local, __LINK_PROGRAM_ERR);
}

cl_program prog = ::clLinkProgram(
ctx(),
0,
NULL,
options,
2,
programs,
notifyFptr,
data,
&error_local);

detail::errHandler(error_local,__COMPILE_PROGRAM_ERR);
if (err != NULL) {
*err = error_local;
}

return Program(prog);
}

inline Program linkProgram(
vector<Program> inputPrograms,
const char* options = NULL,
void (CL_CALLBACK * notifyFptr)(cl_program, void *) = NULL,
void* data = NULL,
cl_int* err = NULL) 
{
cl_int error_local = CL_SUCCESS;

vector<cl_program> programs(inputPrograms.size());

for (unsigned int i = 0; i < inputPrograms.size(); i++) {
programs[i] = inputPrograms[i]();
}

Context ctx;
if(inputPrograms.size() > 0) {
ctx = inputPrograms[0].getInfo<CL_PROGRAM_CONTEXT>(&error_local);
if(error_local!=CL_SUCCESS) {
detail::errHandler(error_local, __LINK_PROGRAM_ERR);
}
}
cl_program prog = ::clLinkProgram(
ctx(),
0,
NULL,
options,
(cl_uint)inputPrograms.size(),
programs.data(),
notifyFptr,
data,
&error_local);

detail::errHandler(error_local,__COMPILE_PROGRAM_ERR);
if (err != NULL) {
*err = error_local;
}

return Program(prog, false);
}
#endif 

template <>
inline cl_int cl::Program::getInfo(cl_program_info name, vector<vector<unsigned char>>* param) const
{
if (name != CL_PROGRAM_BINARIES) {
return CL_INVALID_VALUE;
}
if (param) {

vector<size_type> sizes = getInfo<CL_PROGRAM_BINARY_SIZES>();
size_type numBinaries = sizes.size();

param->resize(numBinaries);
for (size_type i = 0; i < numBinaries; ++i) {
(*param)[i].resize(sizes[i]);
}

return detail::errHandler(
detail::getInfo(&::clGetProgramInfo, object_, name, param),
__GET_PROGRAM_INFO_ERR);
}

return CL_SUCCESS;
}

template<>
inline vector<vector<unsigned char>> cl::Program::getInfo<CL_PROGRAM_BINARIES>(cl_int* err) const
{
vector<vector<unsigned char>> binariesVectors;

cl_int result = getInfo(CL_PROGRAM_BINARIES, &binariesVectors);
if (err != NULL) {
*err = result;
}
return binariesVectors;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 220
template <>
inline cl_int cl::Program::setSpecializationConstant(cl_uint index, const bool &value)
{
cl_uchar ucValue = value ? CL_UCHAR_MAX : 0;
return detail::errHandler(
::clSetProgramSpecializationConstant(
object_,
index,
sizeof(ucValue),
&ucValue),
__SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
}
#endif 

inline Kernel::Kernel(const Program& program, const char* name, cl_int* err)
{
cl_int error;

object_ = ::clCreateKernel(program(), name, &error);
detail::errHandler(error, __CREATE_KERNEL_ERR);

if (err != NULL) {
*err = error;
}

}

enum class QueueProperties : cl_command_queue_properties
{
None = 0,
Profiling = CL_QUEUE_PROFILING_ENABLE,
OutOfOrder = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
};

inline QueueProperties operator|(QueueProperties lhs, QueueProperties rhs)
{
return static_cast<QueueProperties>(static_cast<cl_command_queue_properties>(lhs) | static_cast<cl_command_queue_properties>(rhs));
}

inline QueueProperties operator&(QueueProperties lhs, QueueProperties rhs)
{
return static_cast<QueueProperties>(static_cast<cl_command_queue_properties>(lhs) & static_cast<cl_command_queue_properties>(rhs));
}


class CommandQueue : public detail::Wrapper<cl_command_queue>
{
private:
static std::once_flag default_initialized_;
static CommandQueue default_;
static cl_int default_error_;


static void makeDefault()
{

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
try
#endif
{
int error;
Context context = Context::getDefault(&error);

if (error != CL_SUCCESS) {
default_error_ = error;
}
else {
Device device = Device::getDefault();
default_ = CommandQueue(context, device, 0, &default_error_);
}
}
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
catch (cl::Error &e) {
default_error_ = e.err();
}
#endif
}


static void makeDefaultProvided(const CommandQueue &c) {
default_ = c;
}

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE

static void unitTestClearDefault() {
default_ = CommandQueue();
}
#endif 



CommandQueue(
cl_command_queue_properties properties,
cl_int* err = NULL)
{
cl_int error;

Context context = Context::getDefault(&error);
detail::errHandler(error, __CREATE_CONTEXT_ERR);

if (error != CL_SUCCESS) {
if (err != NULL) {
*err = error;
}
}
else {
Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties, 0 };
if ((properties & CL_QUEUE_ON_DEVICE) == 0) {
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);
}
else {
error = CL_INVALID_QUEUE_PROPERTIES;
}

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), device(), properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}
}


CommandQueue(
QueueProperties properties,
cl_int* err = NULL)
{
cl_int error;

Context context = Context::getDefault(&error);
detail::errHandler(error, __CREATE_CONTEXT_ERR);

if (error != CL_SUCCESS) {
if (err != NULL) {
*err = error;
}
}
else {
Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };

object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), device(), static_cast<cl_command_queue_properties>(properties), &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 

}
}


explicit CommandQueue(
const Context& context,
cl_command_queue_properties properties = 0,
cl_int* err = NULL)
{
cl_int error;
bool useWithProperties;
vector<cl::Device> devices;
error = context.getInfo(CL_CONTEXT_DEVICES, &devices);

detail::errHandler(error, __CREATE_CONTEXT_ERR);

if (error != CL_SUCCESS)
{
if (err != NULL) {
*err = error;
}
return;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties, 0 };
if ((properties & CL_QUEUE_ON_DEVICE) == 0) {
object_ = ::clCreateCommandQueueWithProperties(
context(), devices[0](), queue_properties, &error);
}
else {
error = CL_INVALID_QUEUE_PROPERTIES;
}

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), devices[0](), properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}


explicit CommandQueue(
const Context& context,
QueueProperties properties,
cl_int* err = NULL)
{
cl_int error;
bool useWithProperties;
vector<cl::Device> devices;
error = context.getInfo(CL_CONTEXT_DEVICES, &devices);

detail::errHandler(error, __CREATE_CONTEXT_ERR);

if (error != CL_SUCCESS)
{
if (err != NULL) {
*err = error;
}
return;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), devices[0](), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), devices[0](), static_cast<cl_command_queue_properties>(properties), &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}


CommandQueue(
const Context& context,
const Device& device,
cl_command_queue_properties properties = 0,
cl_int* err = NULL)
{
cl_int error;
bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties, 0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), device(), properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}


CommandQueue(
const Context& context,
const Device& device,
QueueProperties properties,
cl_int* err = NULL)
{
cl_int error;
bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
{
cl_uint version = detail::getContextPlatformVersion(context());
useWithProperties = (version >= 0x20000); 
}
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
useWithProperties = true;
#else
useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
if (useWithProperties) {
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
if (!useWithProperties) {
object_ = ::clCreateCommandQueue(
context(), device(), static_cast<cl_command_queue_properties>(properties), &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
}
#endif 
}

static CommandQueue getDefault(cl_int * err = NULL) 
{
std::call_once(default_initialized_, makeDefault);
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
detail::errHandler(default_error_, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
#else 
detail::errHandler(default_error_, __CREATE_COMMAND_QUEUE_ERR);
#endif 
if (err != NULL) {
*err = default_error_;
}
return default_;
}


static CommandQueue setDefault(const CommandQueue &default_queue)
{
std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_queue));
detail::errHandler(default_error_);
return default_;
}

CommandQueue() { }



explicit CommandQueue(const cl_command_queue& commandQueue, bool retainObject = false) : 
detail::Wrapper<cl_type>(commandQueue, retainObject) { }

CommandQueue& operator = (const cl_command_queue& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


CommandQueue(const CommandQueue& queue) : detail::Wrapper<cl_type>(queue) {}


CommandQueue& operator = (const CommandQueue &queue)
{
detail::Wrapper<cl_type>::operator=(queue);
return *this;
}


CommandQueue(CommandQueue&& queue) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(queue)) {}


CommandQueue& operator = (CommandQueue &&queue)
{
detail::Wrapper<cl_type>::operator=(std::move(queue));
return *this;
}

template <typename T>
cl_int getInfo(cl_command_queue_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(
&::clGetCommandQueueInfo, object_, name, param),
__GET_COMMAND_QUEUE_INFO_ERR);
}

template <cl_command_queue_info name> typename
detail::param_traits<detail::cl_command_queue_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_command_queue_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}

cl_int enqueueReadBuffer(
const Buffer& buffer,
cl_bool blocking,
size_type offset,
size_type size,
void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueReadBuffer(
object_, buffer(), blocking, offset, size,
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_READ_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueWriteBuffer(
const Buffer& buffer,
cl_bool blocking,
size_type offset,
size_type size,
const void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueWriteBuffer(
object_, buffer(), blocking, offset, size,
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_WRITE_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueCopyBuffer(
const Buffer& src,
const Buffer& dst,
size_type src_offset,
size_type dst_offset,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueCopyBuffer(
object_, src(), dst(), src_offset, dst_offset, size,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQEUE_COPY_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
cl_int enqueueReadBufferRect(
const Buffer& buffer,
cl_bool blocking,
const array<size_type, 3>& buffer_offset,
const array<size_type, 3>& host_offset,
const array<size_type, 3>& region,
size_type buffer_row_pitch,
size_type buffer_slice_pitch,
size_type host_row_pitch,
size_type host_slice_pitch,
void *ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueReadBufferRect(
object_, 
buffer(), 
blocking,
buffer_offset.data(),
host_offset.data(),
region.data(),
buffer_row_pitch,
buffer_slice_pitch,
host_row_pitch,
host_slice_pitch,
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_READ_BUFFER_RECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueWriteBufferRect(
const Buffer& buffer,
cl_bool blocking,
const array<size_type, 3>& buffer_offset,
const array<size_type, 3>& host_offset,
const array<size_type, 3>& region,
size_type buffer_row_pitch,
size_type buffer_slice_pitch,
size_type host_row_pitch,
size_type host_slice_pitch,
const void *ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueWriteBufferRect(
object_, 
buffer(), 
blocking,
buffer_offset.data(),
host_offset.data(),
region.data(),
buffer_row_pitch,
buffer_slice_pitch,
host_row_pitch,
host_slice_pitch,
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_WRITE_BUFFER_RECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueCopyBufferRect(
const Buffer& src,
const Buffer& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
size_type src_row_pitch,
size_type src_slice_pitch,
size_type dst_row_pitch,
size_type dst_slice_pitch,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueCopyBufferRect(
object_, 
src(), 
dst(), 
src_origin.data(),
dst_origin.data(),
region.data(),
src_row_pitch,
src_slice_pitch,
dst_row_pitch,
dst_slice_pitch,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQEUE_COPY_BUFFER_RECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 120

template<typename PatternType>
cl_int enqueueFillBuffer(
const Buffer& buffer,
PatternType pattern,
size_type offset,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueFillBuffer(
object_, 
buffer(),
static_cast<void*>(&pattern),
sizeof(PatternType), 
offset, 
size,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_FILL_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 

cl_int enqueueReadImage(
const Image& image,
cl_bool blocking,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
size_type row_pitch,
size_type slice_pitch,
void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueReadImage(
object_, 
image(), 
blocking, 
origin.data(),
region.data(), 
row_pitch, 
slice_pitch, 
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_READ_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueWriteImage(
const Image& image,
cl_bool blocking,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
size_type row_pitch,
size_type slice_pitch,
const void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueWriteImage(
object_, 
image(), 
blocking, 
origin.data(),
region.data(), 
row_pitch, 
slice_pitch, 
ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_WRITE_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueCopyImage(
const Image& src,
const Image& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueCopyImage(
object_, 
src(), 
dst(), 
src_origin.data(),
dst_origin.data(), 
region.data(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_COPY_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

cl_int enqueueFillImage(
const Image& image,
cl_float4 fillColor,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueFillImage(
object_, 
image(),
static_cast<void*>(&fillColor), 
origin.data(),
region.data(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_FILL_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


cl_int enqueueFillImage(
const Image& image,
cl_int4 fillColor,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueFillImage(
object_, 
image(),
static_cast<void*>(&fillColor), 
origin.data(),
region.data(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_FILL_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


cl_int enqueueFillImage(
const Image& image,
cl_uint4 fillColor,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueFillImage(
object_, 
image(),
static_cast<void*>(&fillColor), 
origin.data(),
region.data(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_FILL_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 

cl_int enqueueCopyImageToBuffer(
const Image& src,
const Buffer& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& region,
size_type dst_offset,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueCopyImageToBuffer(
object_, 
src(), 
dst(), 
src_origin.data(),
region.data(), 
dst_offset,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueCopyBufferToImage(
const Buffer& src,
const Image& dst,
size_type src_offset,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueCopyBufferToImage(
object_, 
src(), 
dst(), 
src_offset,
dst_origin.data(), 
region.data(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

void* enqueueMapBuffer(
const Buffer& buffer,
cl_bool blocking,
cl_map_flags flags,
size_type offset,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL,
cl_int* err = NULL) const
{
cl_event tmp;
cl_int error;
void * result = ::clEnqueueMapBuffer(
object_, buffer(), blocking, flags, offset, size,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL,
&error);

detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
if (event != NULL && error == CL_SUCCESS)
*event = tmp;

return result;
}

void* enqueueMapImage(
const Image& buffer,
cl_bool blocking,
cl_map_flags flags,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
size_type * row_pitch,
size_type * slice_pitch,
const vector<Event>* events = NULL,
Event* event = NULL,
cl_int* err = NULL) const
{
cl_event tmp;
cl_int error;
void * result = ::clEnqueueMapImage(
object_, buffer(), blocking, flags,
origin.data(), 
region.data(),
row_pitch, slice_pitch,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL,
&error);

detail::errHandler(error, __ENQUEUE_MAP_IMAGE_ERR);
if (err != NULL) {
*err = error;
}
if (event != NULL && error == CL_SUCCESS)
*event = tmp;
return result;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T>
cl_int enqueueMapSVM(
T* ptr,
cl_bool blocking,
cl_map_flags flags,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(::clEnqueueSVMMap(
object_, blocking, flags, static_cast<void*>(ptr), size,
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MAP_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}



template<typename T, class D>
cl_int enqueueMapSVM(
cl::pointer<T, D> &ptr,
cl_bool blocking,
cl_map_flags flags,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(::clEnqueueSVMMap(
object_, blocking, flags, static_cast<void*>(ptr.get()), size,
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MAP_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


template<typename T, class Alloc>
cl_int enqueueMapSVM(
cl::vector<T, Alloc> &container,
cl_bool blocking,
cl_map_flags flags,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(::clEnqueueSVMMap(
object_, blocking, flags, static_cast<void*>(container.data()), container.size()*sizeof(T),
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MAP_BUFFER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 

cl_int enqueueUnmapMemObject(
const Memory& memory,
void* mapped_ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueUnmapMemObject(
object_, memory(), mapped_ptr,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T>
cl_int enqueueUnmapSVM(
T* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueSVMUnmap(
object_, static_cast<void*>(ptr),
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


template<typename T, class D>
cl_int enqueueUnmapSVM(
cl::pointer<T, D> &ptr,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueSVMUnmap(
object_, static_cast<void*>(ptr.get()),
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


template<typename T, class Alloc>
cl_int enqueueUnmapSVM(
cl::vector<T, Alloc> &container,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueSVMUnmap(
object_, static_cast<void*>(container.data()),
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

cl_int enqueueMarkerWithWaitList(
const vector<Event> *events = 0,
Event *event = 0) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueMarkerWithWaitList(
object_,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MARKER_WAIT_LIST_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


cl_int enqueueBarrierWithWaitList(
const vector<Event> *events = 0,
Event *event = 0) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueBarrierWithWaitList(
object_,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_BARRIER_WAIT_LIST_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


cl_int enqueueMigrateMemObjects(
const vector<Memory> &memObjects,
cl_mem_migration_flags flags,
const vector<Event>* events = NULL,
Event* event = NULL
) const
{
cl_event tmp;

vector<cl_mem> localMemObjects(memObjects.size());

for( int i = 0; i < (int)memObjects.size(); ++i ) {
localMemObjects[i] = memObjects[i]();
}

cl_int err = detail::errHandler(
::clEnqueueMigrateMemObjects(
object_, 
(cl_uint)memObjects.size(), 
localMemObjects.data(),
flags,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 


#if CL_HPP_TARGET_OPENCL_VERSION >= 210

template<typename T>
cl_int enqueueMigrateSVM(
const cl::vector<T*> &svmRawPointers,
const cl::vector<size_type> &sizes,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(::clEnqueueSVMMigrateMem(
object_,
svmRawPointers.size(), static_cast<void**>(svmRawPointers.data()),
sizes.data(), 
flags,
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MIGRATE_SVM_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


template<typename T>
cl_int enqueueMigrateSVM(
const cl::vector<T*> &svmRawPointers,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
return enqueueMigrateSVM(svmRawPointers, cl::vector<size_type>(svmRawPointers.size()), flags, events, event);
}



template<typename T, class D>
cl_int enqueueMigrateSVM(
const cl::vector<cl::pointer<T, D>> &svmPointers,
const cl::vector<size_type> &sizes,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl::vector<void*> svmRawPointers;
svmRawPointers.reserve(svmPointers.size());
for (auto p : svmPointers) {
svmRawPointers.push_back(static_cast<void*>(p.get()));
}

return enqueueMigrateSVM(svmRawPointers, sizes, flags, events, event);
}



template<typename T, class D>
cl_int enqueueMigrateSVM(
const cl::vector<cl::pointer<T, D>> &svmPointers,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
return enqueueMigrateSVM(svmPointers, cl::vector<size_type>(svmPointers.size()), flags, events, event);
}


template<typename T, class Alloc>
cl_int enqueueMigrateSVM(
const cl::vector<cl::vector<T, Alloc>> &svmContainers,
const cl::vector<size_type> &sizes,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl::vector<void*> svmRawPointers;
svmRawPointers.reserve(svmContainers.size());
for (auto p : svmContainers) {
svmRawPointers.push_back(static_cast<void*>(p.data()));
}

return enqueueMigrateSVM(svmRawPointers, sizes, flags, events, event);
}


template<typename T, class Alloc>
cl_int enqueueMigrateSVM(
const cl::vector<cl::vector<T, Alloc>> &svmContainers,
cl_mem_migration_flags flags = 0,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
return enqueueMigrateSVM(svmContainers, cl::vector<size_type>(svmContainers.size()), flags, events, event);
}

#endif 

cl_int enqueueNDRangeKernel(
const Kernel& kernel,
const NDRange& offset,
const NDRange& global,
const NDRange& local = NullRange,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueNDRangeKernel(
object_, kernel(), (cl_uint) global.dimensions(),
offset.dimensions() != 0 ? (const size_type*) offset : NULL,
(const size_type*) global,
local.dimensions() != 0 ? (const size_type*) local : NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_NDRANGE_KERNEL_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_int enqueueTask(
const Kernel& kernel,
const vector<Event>* events = NULL,
Event* event = NULL) const CL_API_SUFFIX__VERSION_1_2_DEPRECATED
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueTask(
object_, kernel(),
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_TASK_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif 

cl_int enqueueNativeKernel(
void (CL_CALLBACK *userFptr)(void *),
std::pair<void*, size_type> args,
const vector<Memory>* mem_objects = NULL,
const vector<const void*>* mem_locs = NULL,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
size_type elements = 0;
if (mem_objects != NULL) {
elements = mem_objects->size();
}
vector<cl_mem> mems(elements);
for (unsigned int i = 0; i < elements; i++) {
mems[i] = ((*mem_objects)[i])();
}

cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueNativeKernel(
object_, userFptr, args.first, args.second,
(mem_objects != NULL) ? (cl_uint) mem_objects->size() : 0,
mems.data(),
(mem_locs != NULL && mem_locs->size() > 0) ? (const void **) &mem_locs->front() : NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_NATIVE_KERNEL);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}


#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
CL_API_PREFIX__VERSION_1_1_DEPRECATED 
cl_int enqueueMarker(Event* event = NULL) const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueMarker(
object_, 
(event != NULL) ? &tmp : NULL),
__ENQUEUE_MARKER_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

CL_API_PREFIX__VERSION_1_1_DEPRECATED
cl_int enqueueWaitForEvents(const vector<Event>& events) const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
{
return detail::errHandler(
::clEnqueueWaitForEvents(
object_,
(cl_uint) events.size(),
events.size() > 0 ? (const cl_event*) &events.front() : NULL),
__ENQUEUE_WAIT_FOR_EVENTS_ERR);
}
#endif 

cl_int enqueueAcquireGLObjects(
const vector<Memory>* mem_objects = NULL,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueAcquireGLObjects(
object_,
(mem_objects != NULL) ? (cl_uint) mem_objects->size() : 0,
(mem_objects != NULL && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_ACQUIRE_GL_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueReleaseGLObjects(
const vector<Memory>* mem_objects = NULL,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueReleaseGLObjects(
object_,
(mem_objects != NULL) ? (cl_uint) mem_objects->size() : 0,
(mem_objects != NULL && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_RELEASE_GL_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

#if defined (CL_HPP_USE_DX_INTEROP)
typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clEnqueueAcquireD3D10ObjectsKHR)(
cl_command_queue command_queue, cl_uint num_objects,
const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
const cl_event* event_wait_list, cl_event* event);
typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clEnqueueReleaseD3D10ObjectsKHR)(
cl_command_queue command_queue, cl_uint num_objects,
const cl_mem* mem_objects,  cl_uint num_events_in_wait_list,
const cl_event* event_wait_list, cl_event* event);

cl_int enqueueAcquireD3D10Objects(
const vector<Memory>* mem_objects = NULL,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
static PFN_clEnqueueAcquireD3D10ObjectsKHR pfn_clEnqueueAcquireD3D10ObjectsKHR = NULL;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
cl_context context = getInfo<CL_QUEUE_CONTEXT>();
cl::Device device(getInfo<CL_QUEUE_DEVICE>());
cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>();
CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueAcquireD3D10ObjectsKHR);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueAcquireD3D10ObjectsKHR);
#endif

cl_event tmp;
cl_int err = detail::errHandler(
pfn_clEnqueueAcquireD3D10ObjectsKHR(
object_,
(mem_objects != NULL) ? (cl_uint) mem_objects->size() : 0,
(mem_objects != NULL && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_ACQUIRE_GL_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

cl_int enqueueReleaseD3D10Objects(
const vector<Memory>* mem_objects = NULL,
const vector<Event>* events = NULL,
Event* event = NULL) const
{
static PFN_clEnqueueReleaseD3D10ObjectsKHR pfn_clEnqueueReleaseD3D10ObjectsKHR = NULL;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
cl_context context = getInfo<CL_QUEUE_CONTEXT>();
cl::Device device(getInfo<CL_QUEUE_DEVICE>());
cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>();
CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueReleaseD3D10ObjectsKHR);
#endif 
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueReleaseD3D10ObjectsKHR);
#endif 

cl_event tmp;
cl_int err = detail::errHandler(
pfn_clEnqueueReleaseD3D10ObjectsKHR(
object_,
(mem_objects != NULL) ? (cl_uint) mem_objects->size() : 0,
(mem_objects != NULL && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): NULL,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_RELEASE_GL_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}
#endif


#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
CL_API_PREFIX__VERSION_1_1_DEPRECATED
cl_int enqueueBarrier() const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
{
return detail::errHandler(
::clEnqueueBarrier(object_),
__ENQUEUE_BARRIER_ERR);
}
#endif 

cl_int flush() const
{
return detail::errHandler(::clFlush(object_), __FLUSH_ERR);
}

cl_int finish() const
{
return detail::errHandler(::clFinish(object_), __FINISH_ERR);
}
}; 

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag CommandQueue::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ CommandQueue CommandQueue::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int CommandQueue::default_error_ = CL_SUCCESS;


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
enum class DeviceQueueProperties : cl_command_queue_properties
{
None = 0,
Profiling = CL_QUEUE_PROFILING_ENABLE,
};

inline DeviceQueueProperties operator|(DeviceQueueProperties lhs, DeviceQueueProperties rhs)
{
return static_cast<DeviceQueueProperties>(static_cast<cl_command_queue_properties>(lhs) | static_cast<cl_command_queue_properties>(rhs));
}


class DeviceCommandQueue : public detail::Wrapper<cl_command_queue>
{
public:


DeviceCommandQueue() { }


DeviceCommandQueue(DeviceQueueProperties properties, cl_int* err = NULL)
{
cl_int error;
cl::Context context = cl::Context::getDefault();
cl::Device device = cl::Device::getDefault();

cl_command_queue_properties mergedProperties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);

cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, mergedProperties, 0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}


DeviceCommandQueue(
const Context& context,
const Device& device,
DeviceQueueProperties properties = DeviceQueueProperties::None,
cl_int* err = NULL)
{
cl_int error;

cl_command_queue_properties mergedProperties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, mergedProperties, 0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}


DeviceCommandQueue(
const Context& context,
const Device& device,
cl_uint queueSize,
DeviceQueueProperties properties = DeviceQueueProperties::None,
cl_int* err = NULL)
{
cl_int error;

cl_command_queue_properties mergedProperties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, mergedProperties,
CL_QUEUE_SIZE, queueSize, 
0 };
object_ = ::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error);

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}
}


explicit DeviceCommandQueue(const cl_command_queue& commandQueue, bool retainObject = false) :
detail::Wrapper<cl_type>(commandQueue, retainObject) { }

DeviceCommandQueue& operator = (const cl_command_queue& rhs)
{
detail::Wrapper<cl_type>::operator=(rhs);
return *this;
}


DeviceCommandQueue(const DeviceCommandQueue& queue) : detail::Wrapper<cl_type>(queue) {}


DeviceCommandQueue& operator = (const DeviceCommandQueue &queue)
{
detail::Wrapper<cl_type>::operator=(queue);
return *this;
}


DeviceCommandQueue(DeviceCommandQueue&& queue) CL_HPP_NOEXCEPT_ : detail::Wrapper<cl_type>(std::move(queue)) {}


DeviceCommandQueue& operator = (DeviceCommandQueue &&queue)
{
detail::Wrapper<cl_type>::operator=(std::move(queue));
return *this;
}

template <typename T>
cl_int getInfo(cl_command_queue_info name, T* param) const
{
return detail::errHandler(
detail::getInfo(
&::clGetCommandQueueInfo, object_, name, param),
__GET_COMMAND_QUEUE_INFO_ERR);
}

template <cl_command_queue_info name> typename
detail::param_traits<detail::cl_command_queue_info, name>::param_type
getInfo(cl_int* err = NULL) const
{
typename detail::param_traits<
detail::cl_command_queue_info, name>::param_type param;
cl_int result = getInfo(name, &param);
if (err != NULL) {
*err = result;
}
return param;
}


static DeviceCommandQueue makeDefault(
cl_int *err = nullptr)
{
cl_int error;
cl::Context context = cl::Context::getDefault();
cl::Device device = cl::Device::getDefault();

cl_command_queue_properties properties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties,
0 };
DeviceCommandQueue deviceQueue(
::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error));

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}

return deviceQueue;
}


static DeviceCommandQueue makeDefault(
const Context &context, const Device &device, cl_int *err = nullptr)
{
cl_int error;

cl_command_queue_properties properties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties,
0 };
DeviceCommandQueue deviceQueue(
::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error));

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}

return deviceQueue;
}


static DeviceCommandQueue makeDefault(
const Context &context, const Device &device, cl_uint queueSize, cl_int *err = nullptr)
{
cl_int error;

cl_command_queue_properties properties =
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
cl_queue_properties queue_properties[] = {
CL_QUEUE_PROPERTIES, properties,
CL_QUEUE_SIZE, queueSize,
0 };
DeviceCommandQueue deviceQueue(
::clCreateCommandQueueWithProperties(
context(), device(), queue_properties, &error));

detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
if (err != NULL) {
*err = error;
}

return deviceQueue;
}



#if CL_HPP_TARGET_OPENCL_VERSION >= 210

static DeviceCommandQueue updateDefault(const Context &context, const Device &device, const DeviceCommandQueue &default_queue, cl_int *err = nullptr)
{
cl_int error;
error = clSetDefaultDeviceCommandQueue(context.get(), device.get(), default_queue.get());

detail::errHandler(error, __SET_DEFAULT_DEVICE_COMMAND_QUEUE_ERR);
if (err != NULL) {
*err = error;
}
return default_queue;
}


static DeviceCommandQueue getDefault(const CommandQueue &queue, cl_int * err = NULL)
{
return queue.getInfo<CL_QUEUE_DEVICE_DEFAULT>(err);
}

#endif 
}; 

namespace detail
{
template <>
struct KernelArgumentHandler<cl::DeviceCommandQueue, void>
{
static size_type size(const cl::DeviceCommandQueue&) { return sizeof(cl_command_queue); }
static const cl_command_queue* ptr(const cl::DeviceCommandQueue& value) { return &(value()); }
};
} 

#endif 


template< typename IteratorType >
Buffer::Buffer(
const Context &context,
IteratorType startIterator,
IteratorType endIterator,
bool readOnly,
bool useHostPtr,
cl_int* err)
{
typedef typename std::iterator_traits<IteratorType>::value_type DataType;
cl_int error;

cl_mem_flags flags = 0;
if( readOnly ) {
flags |= CL_MEM_READ_ONLY;
}
else {
flags |= CL_MEM_READ_WRITE;
}
if( useHostPtr ) {
flags |= CL_MEM_USE_HOST_PTR;
}

size_type size = sizeof(DataType)*(endIterator - startIterator);

if( useHostPtr ) {
object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
} else {
object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
}

detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}

if( !useHostPtr ) {
CommandQueue queue(context, 0, &error);
detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}

error = cl::copy(queue, startIterator, endIterator, *this);
detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}
}

template< typename IteratorType >
Buffer::Buffer(
const CommandQueue &queue,
IteratorType startIterator,
IteratorType endIterator,
bool readOnly,
bool useHostPtr,
cl_int* err)
{
typedef typename std::iterator_traits<IteratorType>::value_type DataType;
cl_int error;

cl_mem_flags flags = 0;
if (readOnly) {
flags |= CL_MEM_READ_ONLY;
}
else {
flags |= CL_MEM_READ_WRITE;
}
if (useHostPtr) {
flags |= CL_MEM_USE_HOST_PTR;
}

size_type size = sizeof(DataType)*(endIterator - startIterator);

Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

if (useHostPtr) {
object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
}
else {
object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
}

detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}

if (!useHostPtr) {
error = cl::copy(queue, startIterator, endIterator, *this);
detail::errHandler(error, __CREATE_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
}
}

inline cl_int enqueueReadBuffer(
const Buffer& buffer,
cl_bool blocking,
size_type offset,
size_type size,
void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueReadBuffer(buffer, blocking, offset, size, ptr, events, event);
}

inline cl_int enqueueWriteBuffer(
const Buffer& buffer,
cl_bool blocking,
size_type offset,
size_type size,
const void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueWriteBuffer(buffer, blocking, offset, size, ptr, events, event);
}

inline void* enqueueMapBuffer(
const Buffer& buffer,
cl_bool blocking,
cl_map_flags flags,
size_type offset,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL,
cl_int* err = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
if (err != NULL) {
*err = error;
}

void * result = ::clEnqueueMapBuffer(
queue(), buffer(), blocking, flags, offset, size,
(events != NULL) ? (cl_uint) events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*) &events->front() : NULL,
(cl_event*) event,
&error);

detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
if (err != NULL) {
*err = error;
}
return result;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T>
inline cl_int enqueueMapSVM(
T* ptr,
cl_bool blocking,
cl_map_flags flags,
size_type size,
const vector<Event>* events,
Event* event)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
}

return queue.enqueueMapSVM(
ptr, blocking, flags, size, events, event);
}


template<typename T, class D>
inline cl_int enqueueMapSVM(
cl::pointer<T, D> &ptr,
cl_bool blocking,
cl_map_flags flags,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
}

return queue.enqueueMapSVM(
ptr, blocking, flags, size, events, event);
}


template<typename T, class Alloc>
inline cl_int enqueueMapSVM(
cl::vector<T, Alloc> &container,
cl_bool blocking,
cl_map_flags flags,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
}

return queue.enqueueMapSVM(
container, blocking, flags, events, event);
}

#endif 

inline cl_int enqueueUnmapMemObject(
const Memory& memory,
void* mapped_ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
if (error != CL_SUCCESS) {
return error;
}

cl_event tmp;
cl_int err = detail::errHandler(
::clEnqueueUnmapMemObject(
queue(), memory(), mapped_ptr,
(events != NULL) ? (cl_uint)events->size() : 0,
(events != NULL && events->size() > 0) ? (cl_event*)&events->front() : NULL,
(event != NULL) ? &tmp : NULL),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

if (event != NULL && err == CL_SUCCESS)
*event = tmp;

return err;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T>
inline cl_int enqueueUnmapSVM(
T* ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_UNMAP_MEM_OBJECT_ERR);
}

return detail::errHandler(queue.enqueueUnmapSVM(ptr, events, event), 
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);

}


template<typename T, class D>
inline cl_int enqueueUnmapSVM(
cl::pointer<T, D> &ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_UNMAP_MEM_OBJECT_ERR);
}

return detail::errHandler(queue.enqueueUnmapSVM(ptr, events, event),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);
}


template<typename T, class Alloc>
inline cl_int enqueueUnmapSVM(
cl::vector<T, Alloc> &container,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS) {
return detail::errHandler(error, __ENQUEUE_UNMAP_MEM_OBJECT_ERR);
}

return detail::errHandler(queue.enqueueUnmapSVM(container, events, event),
__ENQUEUE_UNMAP_MEM_OBJECT_ERR);
}

#endif 

inline cl_int enqueueCopyBuffer(
const Buffer& src,
const Buffer& dst,
size_type src_offset,
size_type dst_offset,
size_type size,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueCopyBuffer(src, dst, src_offset, dst_offset, size, events, event);
}


template< typename IteratorType >
inline cl_int copy( IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer )
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS)
return error;

return cl::copy(queue, startIterator, endIterator, buffer);
}


template< typename IteratorType >
inline cl_int copy( const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator )
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);
if (error != CL_SUCCESS)
return error;

return cl::copy(queue, buffer, startIterator, endIterator);
}


template< typename IteratorType >
inline cl_int copy( const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer )
{
typedef typename std::iterator_traits<IteratorType>::value_type DataType;
cl_int error;

size_type length = endIterator-startIterator;
size_type byteLength = length*sizeof(DataType);

DataType *pointer = 
static_cast<DataType*>(queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, byteLength, 0, 0, &error));
if( error != CL_SUCCESS ) {
return error;
}
#if defined(_MSC_VER)
std::copy(
startIterator, 
endIterator, 
stdext::checked_array_iterator<DataType*>(
pointer, length));
#else
std::copy(startIterator, endIterator, pointer);
#endif
Event endEvent;
error = queue.enqueueUnmapMemObject(buffer, pointer, 0, &endEvent);
if( error != CL_SUCCESS ) { 
return error;
}
endEvent.wait();
return CL_SUCCESS;
}


template< typename IteratorType >
inline cl_int copy( const CommandQueue &queue, const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator )
{
typedef typename std::iterator_traits<IteratorType>::value_type DataType;
cl_int error;

size_type length = endIterator-startIterator;
size_type byteLength = length*sizeof(DataType);

DataType *pointer = 
static_cast<DataType*>(queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0, byteLength, 0, 0, &error));
if( error != CL_SUCCESS ) {
return error;
}
std::copy(pointer, pointer + length, startIterator);
Event endEvent;
error = queue.enqueueUnmapMemObject(buffer, pointer, 0, &endEvent);
if( error != CL_SUCCESS ) { 
return error;
}
endEvent.wait();
return CL_SUCCESS;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 200

template<typename T, class Alloc>
inline cl_int mapSVM(cl::vector<T, Alloc> &container)
{
return enqueueMapSVM(container, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE);
}


template<typename T, class Alloc>
inline cl_int unmapSVM(cl::vector<T, Alloc> &container)
{
return enqueueUnmapSVM(container);
}

#endif 

#if CL_HPP_TARGET_OPENCL_VERSION >= 110
inline cl_int enqueueReadBufferRect(
const Buffer& buffer,
cl_bool blocking,
const array<size_type, 3>& buffer_offset,
const array<size_type, 3>& host_offset,
const array<size_type, 3>& region,
size_type buffer_row_pitch,
size_type buffer_slice_pitch,
size_type host_row_pitch,
size_type host_slice_pitch,
void *ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueReadBufferRect(
buffer, 
blocking, 
buffer_offset, 
host_offset,
region,
buffer_row_pitch,
buffer_slice_pitch,
host_row_pitch,
host_slice_pitch,
ptr, 
events, 
event);
}

inline cl_int enqueueWriteBufferRect(
const Buffer& buffer,
cl_bool blocking,
const array<size_type, 3>& buffer_offset,
const array<size_type, 3>& host_offset,
const array<size_type, 3>& region,
size_type buffer_row_pitch,
size_type buffer_slice_pitch,
size_type host_row_pitch,
size_type host_slice_pitch,
const void *ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueWriteBufferRect(
buffer, 
blocking, 
buffer_offset, 
host_offset,
region,
buffer_row_pitch,
buffer_slice_pitch,
host_row_pitch,
host_slice_pitch,
ptr, 
events, 
event);
}

inline cl_int enqueueCopyBufferRect(
const Buffer& src,
const Buffer& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
size_type src_row_pitch,
size_type src_slice_pitch,
size_type dst_row_pitch,
size_type dst_slice_pitch,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueCopyBufferRect(
src,
dst,
src_origin,
dst_origin,
region,
src_row_pitch,
src_slice_pitch,
dst_row_pitch,
dst_slice_pitch,
events, 
event);
}
#endif 

inline cl_int enqueueReadImage(
const Image& image,
cl_bool blocking,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
size_type row_pitch,
size_type slice_pitch,
void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL) 
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueReadImage(
image,
blocking,
origin,
region,
row_pitch,
slice_pitch,
ptr,
events, 
event);
}

inline cl_int enqueueWriteImage(
const Image& image,
cl_bool blocking,
const array<size_type, 3>& origin,
const array<size_type, 3>& region,
size_type row_pitch,
size_type slice_pitch,
const void* ptr,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueWriteImage(
image,
blocking,
origin,
region,
row_pitch,
slice_pitch,
ptr,
events, 
event);
}

inline cl_int enqueueCopyImage(
const Image& src,
const Image& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueCopyImage(
src,
dst,
src_origin,
dst_origin,
region,
events,
event);
}

inline cl_int enqueueCopyImageToBuffer(
const Image& src,
const Buffer& dst,
const array<size_type, 3>& src_origin,
const array<size_type, 3>& region,
size_type dst_offset,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueCopyImageToBuffer(
src,
dst,
src_origin,
region,
dst_offset,
events,
event);
}

inline cl_int enqueueCopyBufferToImage(
const Buffer& src,
const Image& dst,
size_type src_offset,
const array<size_type, 3>& dst_origin,
const array<size_type, 3>& region,
const vector<Event>* events = NULL,
Event* event = NULL)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.enqueueCopyBufferToImage(
src,
dst,
src_offset,
dst_origin,
region,
events,
event);
}


inline cl_int flush(void)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
}

return queue.flush();
}

inline cl_int finish(void)
{
cl_int error;
CommandQueue queue = CommandQueue::getDefault(&error);

if (error != CL_SUCCESS) {
return error;
} 


return queue.finish();
}

class EnqueueArgs
{
private:
CommandQueue queue_;
const NDRange offset_;
const NDRange global_;
const NDRange local_;
vector<Event> events_;

template<typename... Ts>
friend class KernelFunctor;

public:
EnqueueArgs(NDRange global) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(NullRange)
{

}

EnqueueArgs(NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(local)
{

}

EnqueueArgs(NDRange offset, NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(offset), 
global_(global),
local_(local)
{

}

EnqueueArgs(Event e, NDRange global) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(NullRange)
{
events_.push_back(e);
}

EnqueueArgs(Event e, NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(local)
{
events_.push_back(e);
}

EnqueueArgs(Event e, NDRange offset, NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(offset), 
global_(global),
local_(local)
{
events_.push_back(e);
}

EnqueueArgs(const vector<Event> &events, NDRange global) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(NullRange),
events_(events)
{

}

EnqueueArgs(const vector<Event> &events, NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(NullRange), 
global_(global),
local_(local),
events_(events)
{

}

EnqueueArgs(const vector<Event> &events, NDRange offset, NDRange global, NDRange local) : 
queue_(CommandQueue::getDefault()),
offset_(offset), 
global_(global),
local_(local),
events_(events)
{

}

EnqueueArgs(CommandQueue &queue, NDRange global) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(NullRange)
{

}

EnqueueArgs(CommandQueue &queue, NDRange global, NDRange local) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(local)
{

}

EnqueueArgs(CommandQueue &queue, NDRange offset, NDRange global, NDRange local) : 
queue_(queue),
offset_(offset), 
global_(global),
local_(local)
{

}

EnqueueArgs(CommandQueue &queue, Event e, NDRange global) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(NullRange)
{
events_.push_back(e);
}

EnqueueArgs(CommandQueue &queue, Event e, NDRange global, NDRange local) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(local)
{
events_.push_back(e);
}

EnqueueArgs(CommandQueue &queue, Event e, NDRange offset, NDRange global, NDRange local) : 
queue_(queue),
offset_(offset), 
global_(global),
local_(local)
{
events_.push_back(e);
}

EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange global) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(NullRange),
events_(events)
{

}

EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange global, NDRange local) : 
queue_(queue),
offset_(NullRange), 
global_(global),
local_(local),
events_(events)
{

}

EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange offset, NDRange global, NDRange local) : 
queue_(queue),
offset_(offset), 
global_(global),
local_(local),
events_(events)
{

}
};





template<typename... Ts>
class KernelFunctor
{
private:
Kernel kernel_;

template<int index, typename T0, typename... T1s>
void setArgs(T0&& t0, T1s&&... t1s)
{
kernel_.setArg(index, t0);
setArgs<index + 1, T1s...>(std::forward<T1s>(t1s)...);
}

template<int index, typename T0>
void setArgs(T0&& t0)
{
kernel_.setArg(index, t0);
}

template<int index>
void setArgs()
{
}


public:
KernelFunctor(Kernel kernel) : kernel_(kernel)
{}

KernelFunctor(
const Program& program,
const string name,
cl_int * err = NULL) :
kernel_(program, name.c_str(), err)
{}

typedef Event result_type;


Event operator() (
const EnqueueArgs& args,
Ts... ts)
{
Event event;
setArgs<0>(std::forward<Ts>(ts)...);

args.queue_.enqueueNDRangeKernel(
kernel_,
args.offset_,
args.global_,
args.local_,
&args.events_,
&event);

return event;
}


Event operator() (
const EnqueueArgs& args,
Ts... ts,
cl_int &error)
{
Event event;
setArgs<0>(std::forward<Ts>(ts)...);

error = args.queue_.enqueueNDRangeKernel(
kernel_,
args.offset_,
args.global_,
args.local_,
&args.events_,
&event);

return event;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
cl_int setSVMPointers(const vector<void*> &pointerList)
{
return kernel_.setSVMPointers(pointerList);
}

template<typename T0, typename... T1s>
cl_int setSVMPointers(const T0 &t0, T1s &... ts)
{
return kernel_.setSVMPointers(t0, ts...);
}
#endif 

Kernel getKernel()
{
return kernel_;
}
};

namespace compatibility {

template<typename... Ts>
struct make_kernel
{
typedef KernelFunctor<Ts...> FunctorType;

FunctorType functor_;

make_kernel(
const Program& program,
const string name,
cl_int * err = NULL) :
functor_(FunctorType(program, name, err))
{}

make_kernel(
const Kernel kernel) :
functor_(FunctorType(kernel))
{}

typedef Event result_type;

typedef Event type_(
const EnqueueArgs&,
Ts...);

Event operator()(
const EnqueueArgs& enqueueArgs,
Ts... args)
{
return functor_(
enqueueArgs, args...);
}
};
} 



#undef CL_HPP_ERR_STR_
#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS)
#undef __GET_DEVICE_INFO_ERR               
#undef __GET_PLATFORM_INFO_ERR             
#undef __GET_DEVICE_IDS_ERR                
#undef __GET_PLATFORM_IDS_ERR              
#undef __GET_CONTEXT_INFO_ERR              
#undef __GET_EVENT_INFO_ERR                
#undef __GET_EVENT_PROFILE_INFO_ERR        
#undef __GET_MEM_OBJECT_INFO_ERR           
#undef __GET_IMAGE_INFO_ERR                
#undef __GET_SAMPLER_INFO_ERR              
#undef __GET_KERNEL_INFO_ERR               
#undef __GET_KERNEL_ARG_INFO_ERR           
#undef __GET_KERNEL_SUB_GROUP_INFO_ERR     
#undef __GET_KERNEL_WORK_GROUP_INFO_ERR    
#undef __GET_PROGRAM_INFO_ERR              
#undef __GET_PROGRAM_BUILD_INFO_ERR        
#undef __GET_COMMAND_QUEUE_INFO_ERR        
#undef __CREATE_CONTEXT_ERR                
#undef __CREATE_CONTEXT_FROM_TYPE_ERR      
#undef __GET_SUPPORTED_IMAGE_FORMATS_ERR   
#undef __CREATE_BUFFER_ERR                 
#undef __COPY_ERR                          
#undef __CREATE_SUBBUFFER_ERR              
#undef __CREATE_GL_BUFFER_ERR              
#undef __CREATE_GL_RENDER_BUFFER_ERR       
#undef __GET_GL_OBJECT_INFO_ERR            
#undef __CREATE_IMAGE_ERR                  
#undef __CREATE_GL_TEXTURE_ERR             
#undef __IMAGE_DIMENSION_ERR               
#undef __SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR 
#undef __CREATE_USER_EVENT_ERR             
#undef __SET_USER_EVENT_STATUS_ERR         
#undef __SET_EVENT_CALLBACK_ERR            
#undef __WAIT_FOR_EVENTS_ERR               
#undef __CREATE_KERNEL_ERR                 
#undef __SET_KERNEL_ARGS_ERR               
#undef __CREATE_PROGRAM_WITH_SOURCE_ERR    
#undef __CREATE_PROGRAM_WITH_IL_ERR        
#undef __CREATE_PROGRAM_WITH_BINARY_ERR    
#undef __CREATE_PROGRAM_WITH_IL_ERR        
#undef __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR    
#undef __BUILD_PROGRAM_ERR                 
#undef __COMPILE_PROGRAM_ERR               
#undef __LINK_PROGRAM_ERR                  
#undef __CREATE_KERNELS_IN_PROGRAM_ERR     
#undef __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR          
#undef __CREATE_SAMPLER_WITH_PROPERTIES_ERR                
#undef __SET_COMMAND_QUEUE_PROPERTY_ERR    
#undef __ENQUEUE_READ_BUFFER_ERR           
#undef __ENQUEUE_READ_BUFFER_RECT_ERR      
#undef __ENQUEUE_WRITE_BUFFER_ERR          
#undef __ENQUEUE_WRITE_BUFFER_RECT_ERR     
#undef __ENQEUE_COPY_BUFFER_ERR            
#undef __ENQEUE_COPY_BUFFER_RECT_ERR       
#undef __ENQUEUE_FILL_BUFFER_ERR           
#undef __ENQUEUE_READ_IMAGE_ERR            
#undef __ENQUEUE_WRITE_IMAGE_ERR           
#undef __ENQUEUE_COPY_IMAGE_ERR            
#undef __ENQUEUE_FILL_IMAGE_ERR            
#undef __ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR  
#undef __ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR  
#undef __ENQUEUE_MAP_BUFFER_ERR            
#undef __ENQUEUE_MAP_IMAGE_ERR             
#undef __ENQUEUE_UNMAP_MEM_OBJECT_ERR      
#undef __ENQUEUE_NDRANGE_KERNEL_ERR        
#undef __ENQUEUE_NATIVE_KERNEL             
#undef __ENQUEUE_MIGRATE_MEM_OBJECTS_ERR   
#undef __ENQUEUE_MIGRATE_SVM_ERR
#undef __ENQUEUE_ACQUIRE_GL_ERR            
#undef __ENQUEUE_RELEASE_GL_ERR            
#undef __CREATE_PIPE_ERR             
#undef __GET_PIPE_INFO_ERR           
#undef __RETAIN_ERR                        
#undef __RELEASE_ERR                       
#undef __FLUSH_ERR                         
#undef __FINISH_ERR                        
#undef __VECTOR_CAPACITY_ERR               
#undef __CREATE_SUB_DEVICES_ERR            
#undef __CREATE_SUB_DEVICES_ERR            
#undef __ENQUEUE_MARKER_ERR                
#undef __ENQUEUE_WAIT_FOR_EVENTS_ERR       
#undef __ENQUEUE_BARRIER_ERR               
#undef __UNLOAD_COMPILER_ERR               
#undef __CREATE_GL_TEXTURE_2D_ERR          
#undef __CREATE_GL_TEXTURE_3D_ERR          
#undef __CREATE_IMAGE2D_ERR                
#undef __CREATE_IMAGE3D_ERR                
#undef __CREATE_COMMAND_QUEUE_ERR          
#undef __ENQUEUE_TASK_ERR                  
#undef __CREATE_SAMPLER_ERR                
#undef __ENQUEUE_MARKER_WAIT_LIST_ERR                
#undef __ENQUEUE_BARRIER_WAIT_LIST_ERR               
#undef __CLONE_KERNEL_ERR     
#undef __GET_HOST_TIMER_ERR
#undef __GET_DEVICE_AND_HOST_TIMER_ERR

#endif 

#undef CL_HPP_INIT_CL_EXT_FCN_PTR_
#undef CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_

#if defined(CL_HPP_USE_CL_DEVICE_FISSION)
#undef CL_HPP_PARAM_NAME_DEVICE_FISSION_
#endif 

#undef CL_HPP_NOEXCEPT_
#undef CL_HPP_DEFINE_STATIC_MEMBER_

} 

#endif 