

#pragma once

#include <sstream>
#include <memory>

#include "Stream.h"

#include <sycl/sycl.hpp>

#define IMPLEMENTATION_STRING "SYCL 2020"

template <class T>
class SYCLStream : public Stream<T>
{
protected:
size_t array_size;

std::unique_ptr<sycl::queue> queue;

sycl::buffer<T> d_a;
sycl::buffer<T> d_b;
sycl::buffer<T> d_c;
sycl::buffer<T> d_sum;

public:

SYCLStream(const size_t, const int);
~SYCLStream() = default;

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
