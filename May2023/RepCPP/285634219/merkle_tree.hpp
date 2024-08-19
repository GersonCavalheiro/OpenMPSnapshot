#pragma once
#include <cassert>
#include "rescue_prime.hpp"

void
merklize_approach_1(sycl::queue& q,
const sycl::ulong* leaves,
sycl::ulong* const intermediates,
const size_t leaf_count,
const size_t wg_size,
const sycl::ulong4* mds,
const sycl::ulong4* ark1,
const sycl::ulong4* ark2);

