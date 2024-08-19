
#pragma once
#include "targetver.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <limits.h>
#include <string.h>
typedef enum { false = 0, true = 1 } bool;
#define BITS(t) (sizeof(t)*8)
#define SINLINE static inline
typedef enum {
pfdg_success = 0,
pfdg_error_malloc = 1,
pfdg_error_command = 2,
pfdg_error_number_of_args = 3,
pfdg_error_arg = 4,
pfdg_error_arg_value = 5,
pfdg_error_arg_precondition = 6,
pfdg_error_mem_base = 7,
pfdg_error_mem_min = 8,
pfdg_error_mem_chunk_size = 9,
pfdg_error_io = 10
} pfdg_error_t;
#if _M_X64 || __LP64__
#define BITARRAY_WORD uint64_t
#ifdef _MSC_VER
#define POPCNT __popcnt64
#else
#define POPCNT _popcnt64
#endif
#else
#define BITARRAY_WORD uint32_t
#ifdef _MSC_VER
#define POPCNT __popcnt
#else
#define POPCNT _popcnt
#endif
#endif
#ifdef _MSC_VER
#define ALIGN32 __declspec(align(32))
#define MIN __min
#define MAX __max
#define ALIGNED_MALLOC _aligned_malloc
#define ALIGNED_FREE _aligned_free
#define ATOI64 _atoi64
#define _CRT_SECURE_NO_WARNINGS 1
#else
#define ALIGN32 __attribute__((aligned(32)))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ALIGNED_MALLOC(size, alignment) (aligned_alloc((alignment), (size)))
#define ALIGNED_FREE free
#define ATOI64 atoll
#endif
#define NATIVE_ORDER LSB_LITTLE
