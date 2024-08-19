

#pragma once
#ifndef TESTING_BACKEND_HPP
#define TESTING_BACKEND_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

void testing_backend_init_order(void)
{
bool use_acc   = false;
bool omp_aff   = false;
int  dev       = 0;
int  nthreads  = 4;
int  threshold = 20000;

stop_rocalution();

ASSERT_DEATH(set_omp_threads_rocalution(nthreads), ".*Assertion.*");

ASSERT_DEATH(set_omp_threshold_rocalution(threshold), ".*Assertion.*");

set_device_rocalution(device);
init_rocalution();

ASSERT_DEATH(set_omp_affinity_rocalution(omp_aff), ".*Assertion.*");

ASSERT_DEATH(set_device_rocalution(dev), ".*Assertion.*");

ASSERT_DEATH(disable_accelerator_rocalution(use_acc), ".*Assertion.*");

stop_rocalution();
}

void testing_backend(Arguments argus)
{
int  rank         = argus.rank;
int  dev_per_node = argus.dev_per_node;
int  dev          = argus.dev;
int  nthreads     = argus.omp_nthreads;
bool affinity     = argus.omp_affinity;
int  threshold    = argus.omp_threshold;
bool use_acc      = argus.use_acc;

set_device_rocalution(dev);

disable_accelerator_rocalution(use_acc);

set_omp_affinity_rocalution(affinity);

init_rocalution(rank, dev_per_node);

set_omp_threads_rocalution(nthreads);

set_omp_threshold_rocalution(threshold);

stop_rocalution();
}

#endif 
