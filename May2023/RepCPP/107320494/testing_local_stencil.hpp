

#pragma once
#ifndef TESTING_LOCAL_STENCIL_HPP
#define TESTING_LOCAL_STENCIL_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_stencil_bad_args(void)
{
set_device_rocalution(device);
init_rocalution();

LocalStencil<T> stn(Laplace2D);
LocalVector<T>  vec;

{
LocalVector<T>* null_vec = nullptr;
ASSERT_DEATH(stn.Apply(vec, null_vec), ".*Assertion.*out != (NULL|__null)*");
}

{
LocalVector<T>* null_vec = nullptr;
ASSERT_DEATH(stn.ApplyAdd(vec, 1.0, null_vec), ".*Assertion.*out != (NULL|__null)*");
}

stop_rocalution();
}

#endif 
