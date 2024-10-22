

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#if defined(PAGEFAULT)
#define IMPLEMENTATION_STRING "CUDA - Page Fault"
#elif defined(MANAGED)
#define IMPLEMENTATION_STRING "CUDA - Managed Memory"
#else
#define IMPLEMENTATION_STRING "CUDA"
#endif

#define TBSIZE 1024
#define DOT_NUM_BLOCKS 256

template <class T>
class CUDAStream : public Stream<T>
{
protected:
int array_size;

T *sums;

T *d_a;
T *d_b;
T *d_c;
T *d_sum;


public:

CUDAStream(const int, const int);
~CUDAStream();

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
