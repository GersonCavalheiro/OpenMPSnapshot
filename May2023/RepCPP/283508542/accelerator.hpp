#pragma once

#include <iostream>
#include <cstring>
#include <string>
using std::string;

#include "configure.hpp"


#if   PARACABS_USE_ACCELERATOR && PARACABS_USE_CUDA

#include "accelerator_cuda.hpp"

#elif PARACABS_USE_ACCELERATOR && PARACABS_USE_SYCL

#include "accelerator_sycl.hpp"

#else

#define accel

#include "multi_threading/multi_threading.hpp"

namespace paracabs
{
namespace accelerator
{
inline size_t& nblocks()
{
static size_t value = 1;
return value;
}

inline size_t& nthreads()
{
static size_t value = 1;
return value;
}

inline unsigned int nGPUs ()
{
return 0;
}

inline string get_gpu_name (const int i)
{
return "";
}

inline void list_accelerators ()
{
for (unsigned int i = 0; i < nGPUs (); i++)
{
std::cout << get_gpu_name (i) << std::endl;
}
}

inline void synchronize ()
{
return;
}

inline void* malloc (const size_t num)
{
return std::malloc (num);
}

inline void free (void* ptr)
{
std::free (ptr);
}

inline void memcpy_to_accelerator (void* dst, const void* src, const size_t size)
{
std::memcpy (dst, src, size);
}

inline void memcpy_from_accelerator (void* dst, const void* src, const size_t size)
{
std::memcpy (dst, src, size);
}

using AcceleratorThreads = paracabs::multi_threading::HostThreads;
}
}


#define accelerated_for(i, total, ... )    \
{                                          \
threaded_for(i, total, __VA_ARGS__);   \
}


#define accelerated_for_outside_class(i, total, ... )   \
{                                                       \
threaded_for(i, total, __VA_ARGS__);                \
}


#endif

