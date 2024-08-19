
#pragma once

#include <boost/numeric/ublas/opencl.hpp>
#include "../init.hpp"

namespace boost { namespace numeric { namespace ublas { namespace benchmark {

template <typename T>
void init(T &v, unsigned long size, int max_value)
{
}

template <typename T>
void init(vector<T, opencl::storage> &v, unsigned long size, int max_value)
{
}

template <typename T>
void init(matrix<T, opencl::storage> &m, unsigned long size1, unsigned long size2, int max_value)
{
}

template <typename T>
void init(matrix<T, opencl::storage> &m, unsigned long size, int max_value)
{
init(m, size, size, max_value);
}

}}}}
