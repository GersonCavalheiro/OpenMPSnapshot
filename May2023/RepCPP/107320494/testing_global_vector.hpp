

#pragma once
#ifndef TESTING_GLOBAL_VECTOR_HPP
#define TESTING_GLOBAL_VECTOR_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_global_vector_bad_args(void)
{
int safe_size = 100;

set_device_rocalution(device);
init_rocalution();

GlobalVector<T> vec;

{
T* null_data = nullptr;
ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != (NULL|__null)*");
ASSERT_DEATH(vec.SetDataPtr(&null_data, "", safe_size),
".*Assertion.*ptr != (NULL|__null)*");
}

{
T* data = nullptr;
allocate_host(safe_size, &data);
ASSERT_DEATH(vec.LeaveDataPtr(&data), ".*Assertion.*ptr == (NULL|__null)*");
free_host(&data);
}

stop_rocalution();
}

#endif 
