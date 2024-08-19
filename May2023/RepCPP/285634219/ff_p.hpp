#pragma once
#include <CL/sycl.hpp>

inline constexpr uint64_t MOD =
((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

SYCL_EXTERNAL uint64_t
ff_p_add(uint64_t a, uint64_t b);

SYCL_EXTERNAL uint64_t
ff_p_sub(uint64_t a, uint64_t b);

SYCL_EXTERNAL uint64_t
ff_p_mult(uint64_t a, uint64_t b);

SYCL_EXTERNAL uint64_t
ff_p_pow(uint64_t a, const uint64_t b);

SYCL_EXTERNAL uint64_t
ff_p_inv(uint64_t a);

SYCL_EXTERNAL uint64_t
ff_p_div(uint64_t a, uint64_t b);
