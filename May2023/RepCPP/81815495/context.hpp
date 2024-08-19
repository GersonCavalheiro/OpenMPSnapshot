#ifndef VEXCL_BACKEND_OPENCL_CONTEXT_HPP
#define VEXCL_BACKEND_OPENCL_CONTEXT_HPP





#include <vector>
#include <iostream>

#include <vexcl/backend/opencl/defines.hpp>
#include <CL/cl.hpp>

namespace vex {
namespace backend {

namespace opencl {

typedef cl::Context                 context;
typedef cl::Device                  device;
typedef cl::Program                 program;
typedef cl::CommandQueue            command_queue;
typedef cl_command_queue_properties command_queue_properties;
typedef cl_device_id                device_id;
typedef cl::NDRange                 ndrange;


inline void select_context(const command_queue&) {
}

inline device get_device(const command_queue &q) {
return q.getInfo<CL_QUEUE_DEVICE>();
}

inline device_id get_device_id(const command_queue &q) {
return q.getInfo<CL_QUEUE_DEVICE>()();
}

typedef cl_context       context_id;
inline context_id get_context_id(const command_queue &q) {
return q.getInfo<CL_QUEUE_CONTEXT>()();
}

inline context get_context(const command_queue &q) {
return q.getInfo<CL_QUEUE_CONTEXT>();
}

struct compare_contexts {
bool operator()(const context &a, const context &b) const {
return a() < b();
}
};

struct compare_queues {
bool operator()(const command_queue &a, const command_queue &b) const {
return a() < b();
}
};

inline command_queue duplicate_queue(const command_queue &q) {
return command_queue(
q.getInfo<CL_QUEUE_CONTEXT>(), q.getInfo<CL_QUEUE_DEVICE>());
}

inline bool is_cpu(const command_queue &q) {
cl::Device d = q.getInfo<CL_QUEUE_DEVICE>();
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4800)
#endif
return d.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU;
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}


template<class DevFilter>
std::vector<vex::backend::device> device_list(DevFilter&& filter) {
std::vector<cl::Device> device;

std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);

for(auto p = platforms.begin(); p != platforms.end(); p++) {
std::vector<cl::Device> dev_list;

p->getDevices(CL_DEVICE_TYPE_ALL, &dev_list);

for(auto d = dev_list.begin(); d != dev_list.end(); d++) {
if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
if (!filter(*d)) continue;

device.push_back(*d);
}
}

return device;
}


template<class DevFilter>
std::pair<std::vector<cl::Context>, std::vector<command_queue>>
queue_list(DevFilter &&filter, cl_command_queue_properties properties = 0) {
std::vector<cl::Context>      context;
std::vector<command_queue> queue;

std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);

for(auto p = platforms.begin(); p != platforms.end(); p++) {
std::vector<cl::Device> device;
std::vector<cl::Device> dev_list;

p->getDevices(CL_DEVICE_TYPE_ALL, &dev_list);

for(auto d = dev_list.begin(); d != dev_list.end(); d++) {
if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
if (!filter(*d)) continue;

device.push_back(*d);
}

if (device.empty()) continue;

for(auto d = device.begin(); d != device.end(); d++)
try {
context.push_back(cl::Context(std::vector<cl::Device>(1, *d)));
queue.push_back(command_queue(context.back(), *d, properties));
} catch(const cl::Error&) {
}
}

return std::make_pair(context, queue);
}

} 
} 
} 

namespace std {

inline std::ostream& operator<<(std::ostream &os, const vex::backend::opencl::command_queue &q)
{
cl::Device   d(q.getInfo<CL_QUEUE_DEVICE>());
cl::Platform p(d.getInfo<CL_DEVICE_PLATFORM>());

return os << d.getInfo<CL_DEVICE_NAME>()
<< " (" << p.getInfo<CL_PLATFORM_NAME>() << ")";
}

} 

#endif
