

#pragma once
#ifndef TESTING_PARALLEL_MANAGER_HPP
#define TESTING_PARALLEL_MANAGER_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_parallel_manager_bad_args(void)
{
int safe_size = 100;

set_device_rocalution(device);
init_rocalution();

ParallelManager pm;

int* idata = nullptr;
allocate_host(safe_size, &idata);

{
void* null_ptr = nullptr;
ASSERT_DEATH(pm.SetMPICommunicator(null_ptr), ".*Assertion.*comm != (NULL|__null)*");
}

{
int* null_int = nullptr;
ASSERT_DEATH(pm.SetBoundaryIndex(safe_size, null_int),
".*Assertion.*index != (NULL|__null)*");
}

{
int* null_int = nullptr;
ASSERT_DEATH(pm.SetReceivers(safe_size, null_int, idata),
".*Assertion.*recvs != (NULL|__null)*");
ASSERT_DEATH(pm.SetReceivers(safe_size, idata, null_int),
".*Assertion.*recv_offset != (NULL|__null)*");
}

{
int* null_int = nullptr;
ASSERT_DEATH(pm.SetSenders(safe_size, null_int, idata),
".*Assertion.*sends != (NULL|__null)*");
ASSERT_DEATH(pm.SetSenders(safe_size, idata, null_int),
".*Assertion.*send_offset != (NULL|__null)*");
}

free_host(&idata);

stop_rocalution();
}

#endif 
