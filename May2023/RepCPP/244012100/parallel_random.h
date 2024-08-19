#ifndef __PARALLEL_RANDOM_H
#define __PARALLEL_RANDOM_H

#include <iostream>
#include <omp.h>
#include <random>

#include "defs.h"
#include "parallel_array.h"


struct RandomSequence {
const u32 NUM_THREADS;
ParallelArray<u32> arr;

RandomSequence(u32 size, u32 NUM_THREADS = omp_get_max_threads()) : NUM_THREADS(NUM_THREADS), arr(size) {
#pragma omp parallel shared(arr)
{
std::mt19937 gen(std::random_device{}());
#pragma omp for
for (u32 i = 0; i < size; ++i) {
arr[i] = gen();
}
}
}

u32 size() const {
return arr.size();
}

u32 operator[](u32 id) {
return arr[id];
}
};

#endif
