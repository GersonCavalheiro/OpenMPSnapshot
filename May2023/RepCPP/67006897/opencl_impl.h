

#ifndef PROFIT_OPENCL_IMPL_H
#define PROFIT_OPENCL_IMPL_H

#include "profit/opencl.h"

#ifdef PROFIT_OPENCL

# if !defined(PROFIT_OPENCL_MAJOR) || !defined(PROFIT_OPENCL_MINOR)
#  error "No OpenCL version specified"
# elif PROFIT_OPENCL_MAJOR < 1 || (PROFIT_OPENCL_MAJOR == 1 && PROFIT_OPENCL_MINOR < 1 )
#  error "libprofit requires at minimum OpenCL >= 1.1"
# endif


#if defined(__APPLE__) || defined(__MACOSX)
# define CL_SILENCE_DEPRECATION
#endif


# define CL_HPP_ENABLE_EXCEPTIONS


#if !defined(CL_PLATFORM_NOT_FOUND_KHR)
# define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif


# define PASTE(x,y) x ## y ## 0
# define MAKE_VERSION(x,y) PASTE(x,y)
# define CL_HPP_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
# define CL_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
# define CL_HPP_MINIMUM_OPENCL_VERSION PROFIT_OPENCL_TARGET_VERSION


#if defined __GNUC__ && __GNUC__>=6
# pragma GCC diagnostic ignored "-Wignored-attributes"
#elif defined __clang__
# pragma clang diagnostic ignored "-Wmissing-braces"
#endif
#include "profit/cl/cl2.hpp"

namespace profit
{


OpenCL_command_times cl_cmd_times(const cl::Event &evt);

class OpenCLEnvImpl;
typedef std::shared_ptr<OpenCLEnvImpl> OpenCLEnvImplPtr;


class OpenCLEnvImpl : public OpenCLEnv {

public:

OpenCLEnvImpl(cl::Device device, cl_ver_t version, cl::Context context,
cl::CommandQueue queue, cl::Program program,
bool use_double, bool use_profiling) :
use_double(use_double), use_profiling(use_profiling),
device(device), version(version), context(context), queue(queue), program(program)
{ }

static OpenCLEnvImplPtr fromOpenCLEnvPtr(const OpenCLEnvPtr &ptr) {
return std::static_pointer_cast<OpenCLEnvImpl>(ptr);
}


bool use_double;


bool use_profiling;

cl_ver_t get_version() override {
return version;
}

std::string get_platform_name() override {
return cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>();
}

std::string get_device_name() override {
return device.getInfo<CL_DEVICE_NAME>();
}


unsigned long max_local_memory();


unsigned int compute_units();


template <typename T>
cl::Buffer get_buffer(int flags, cl::size_type n_elements) {
return cl::Buffer(context, flags, sizeof(T) * n_elements);
}


cl::Event queue_write(const cl::Buffer &buffer, const void *data, const std::vector<cl::Event>* wait_evts = NULL);


cl::Event queue_kernel(const cl::Kernel &kernel, const cl::NDRange global,
const std::vector<cl::Event>* wait_evts = NULL,
const cl::NDRange &local = cl::NullRange);


cl::Event queue_read(const cl::Buffer &buffer, void *data, const std::vector<cl::Event>* wait_evts = NULL);

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
template <typename PatternType>
cl::Event queue_fill(const cl::Buffer &buffer, PatternType pattern, const std::vector<cl::Event>* wait_evts = NULL) {
cl::Event fevt;
queue.enqueueFillBuffer(buffer, pattern, 0, buffer.getInfo<CL_MEM_SIZE>(), wait_evts, &fevt);
return fevt;
}
#endif


cl::Kernel get_kernel(const std::string &name);

private:


cl::Device device;


cl_ver_t version;


cl::Context context;


cl::CommandQueue queue;


cl::Program program;

};

} 

#endif 

#endif 