
#pragma once

#include <iostream>
#include <stdexcept>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD (data-oriented)"


template <class T>
class STDDataStream : public Stream<T>
{
protected:
int array_size;

std::vector<T> a;
std::vector<T> b;
std::vector<T> c;


public:
STDDataStream(const int, int) noexcept;
~STDDataStream() = default;

virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual void nstream() override;
virtual T dot() override;

virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

