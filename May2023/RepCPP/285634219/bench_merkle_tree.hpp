#pragma once
#include <CL/sycl.hpp>

uint64_t
benchmark_merklize_approach_1(sycl::queue& q,
const size_t leaf_count,
const size_t wg_size);
