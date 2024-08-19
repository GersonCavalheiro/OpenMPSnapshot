

#pragma once

#include "header.h"
#include "timer.h"

namespace trinity { namespace sync {

void reduceTasks(int* array, std::vector<int>* heap, int* count, int stride);
void reduceTasks(int* array, std::vector<int>* heap, int* count, int* off);
void prefixSum(int* values, size_t nb, size_t grain_size);
void reallocBucket(std::vector<int>* bucket, int index, size_t chunk, int verb);


template <typename type_t,
typename = std::enable_if_t<std::is_integral<type_t>::value> >
bool compareAndSwap(type_t* flag, int expected, int value) {
assert(flag != nullptr);
return __sync_bool_compare_and_swap(flag, expected, value);
}


template <typename type_t,
typename = std::enable_if_t<std::is_integral<type_t>::value> >
type_t fetchAndAdd(type_t* shared, int value) {
assert(shared != nullptr);
return __sync_fetch_and_add(shared, value);
}


template <typename type_t,
typename = std::enable_if_t<std::is_integral<type_t>::value> >
type_t fetchAndSub(type_t* shared, int value) {
assert(shared != nullptr);
return __sync_fetch_and_sub(shared, value);
}

}} 