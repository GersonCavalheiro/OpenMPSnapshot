

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <omp.h>

#define IMPLEMENTATION_STRING "OpenMP"

template <class T>
class OMPStream : public Stream<T>
{
protected:
int array_size;

T *a;
T *b;
T *c;

public:
OMPStream(const int, int);
~OMPStream();

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};
