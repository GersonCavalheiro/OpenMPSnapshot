

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "HIP"

template <class T>
class HIPStream : public Stream<T>
{
protected:
int array_size;

T *sums;

T *d_a;
T *d_b;
T *d_c;
T *d_sum;


public:

HIPStream(const int, const int);
~HIPStream();

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
