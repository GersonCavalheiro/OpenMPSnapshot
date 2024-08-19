
#pragma once

#include <iostream>
#include <stdexcept>
#include "RAJA/RAJA.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#ifdef RAJA_TARGET_CPU

typedef RAJA::omp_parallel_for_exec policy;
typedef RAJA::omp_reduce reduce_policy;
#else
const size_t block_size = 128;
typedef RAJA::cuda_exec<block_size> policy;
typedef RAJA::cuda_reduce reduce_policy;
#endif

using RAJA::RangeSegment;


template <class T>
class RAJAStream : public Stream<T>
{
protected:
const int array_size;
const RangeSegment range;

T* d_a;
T* d_b;
T* d_c;

public:

RAJAStream(const int, const int);
~RAJAStream();

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(
std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

