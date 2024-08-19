

#pragma once
#ifndef TESTING_LOCAL_VECTOR_HPP
#define TESTING_LOCAL_VECTOR_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_vector_bad_args(void)
{
int safe_size = 100;

set_device_rocalution(device);
init_rocalution();

LocalVector<T> vec;

{
T* null_ptr = nullptr;
ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != (NULL|__null)*");
ASSERT_DEATH(vec.SetDataPtr(&null_ptr, "", safe_size),
".*Assertion.*ptr != (NULL|__null)*");
}

{
T* vdata = nullptr;
allocate_host(safe_size, &vdata);
ASSERT_DEATH(vec.LeaveDataPtr(&vdata), ".*Assertion.*ptr == (NULL|__null)*");
free_host(&vdata);
}

{
T* null_ptr = nullptr;
ASSERT_DEATH(vec.CopyFromData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
}

{
T* null_ptr = nullptr;
ASSERT_DEATH(vec.CopyToData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
}

{
vec.Allocate("", safe_size);
T* null_T = nullptr;
ASSERT_DEATH(vec.GetContinuousValues(0, safe_size, null_T),
".*Assertion.*values != (NULL|__null)*");
}

{
int* null_int = nullptr;
int* vint     = nullptr;
allocate_host(safe_size, &vint);
ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint),
".*Assertion.*index != (NULL|__null)*");
ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint),
".*Assertion.*size != (NULL|__null)*");
ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int),
".*Assertion.*map != (NULL|__null)*");
free_host(&vint);
}

{
int* null_int = nullptr;
int* vint     = nullptr;
allocate_host(safe_size, &vint);
ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint),
".*Assertion.*index != (NULL|__null)*");
ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint),
".*Assertion.*size != (NULL|__null)*");
ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int),
".*Assertion.*boundary != (NULL|__null)*");
free_host(&vint);
}

stop_rocalution();
}

#endif 
