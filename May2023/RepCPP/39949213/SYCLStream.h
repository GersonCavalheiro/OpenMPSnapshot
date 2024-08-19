

#pragma once

#include <sstream>

#include "Stream.h"
#include "CL/sycl.hpp"

#define IMPLEMENTATION_STRING "SYCL"

namespace sycl_kernels
{
template <class T> class init;
template <class T> class copy;
template <class T> class mul;
template <class T> class add;
template <class T> class triad;
template <class T> class nstream;
template <class T> class dot;
}

template <class T>
class SYCLStream : public Stream<T>
{
protected:
size_t array_size;

cl::sycl::queue *queue;
cl::sycl::buffer<T> *d_a;
cl::sycl::buffer<T> *d_b;
cl::sycl::buffer<T> *d_c;
cl::sycl::buffer<T> *d_sum;

typedef sycl_kernels::init<T> init_kernel;
typedef sycl_kernels::copy<T> copy_kernel;
typedef sycl_kernels::mul<T> mul_kernel;
typedef sycl_kernels::add<T> add_kernel;
typedef sycl_kernels::triad<T> triad_kernel;
typedef sycl_kernels::nstream<T> nstream_kernel;
typedef sycl_kernels::dot<T> dot_kernel;

size_t dot_num_groups;
size_t dot_wgsize;

public:

SYCLStream(const int, const int);
~SYCLStream();

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T    dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

void getDeviceList(void);
