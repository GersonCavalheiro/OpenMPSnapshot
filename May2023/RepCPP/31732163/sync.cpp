

#include "trinity/sync.h"

namespace trinity { namespace sync {

void reduceTasks(int* array, std::vector<int>* heap, int* count, int stride) {
size_t nb = heap->size();
if (nb) {
const int offset = sync::fetchAndAdd(count, int(nb / stride));
std::memcpy(array + (offset * stride), heap->data(), nb * sizeof(int));
heap->clear();
}
#pragma omp barrier
}

namespace {

int kernelPrefixSum(int* begin, int* end, size_t grain_size) {
size_t len = end - begin;
if (len < grain_size) {
for (size_t i = 1; i < len; ++i)
begin[i] += begin[i - 1];
} else {

size_t mid = len / 2;
int sum = 0;

#pragma omp task shared(sum)
sum = kernelPrefixSum(begin, begin + mid, grain_size);
#pragma omp task
kernelPrefixSum(begin + mid, end, grain_size);
#pragma omp taskwait

#pragma omp parallel for
for (size_t i = mid; i < len; ++i)
begin[i] += sum;
}
return begin[len - 1];
}

} 

void prefixSum(int* values, size_t nb, size_t grain_size) {
#pragma omp single
kernelPrefixSum(values, values + nb, grain_size);
}


void reduceTasks(int* array, std::vector<int>* heap, int* count, int* off) {
size_t nb = heap->size();
int tid = omp_get_thread_num();
int cores = omp_get_num_threads();  

if (tid < cores - 1) {
off[tid + 1] = (int) nb;
} else {
off[0] = 0;
*count = (int) nb;
}
#pragma omp barrier

prefixSum(off, (size_t) cores, 10);
std::memcpy(array + off[tid], heap->data(), sizeof(int) * nb);
heap->clear();

#pragma omp single
*count += off[cores - 1]; 
}


void reallocBucket(std::vector<int>* bucket, int index, size_t chunk, int verb) {
if (__builtin_expect(bucket[index].size() <= chunk, 0))
#pragma omp critical(resize)
{
size_t old = bucket[index].size();
if (old <= chunk) {
bucket[index].resize(old * 2);
std::fill(bucket[index].begin() + old, bucket[index].end(), -1);
if (verb)
std::fprintf(stderr, "warning: stenc[%d] was reallocated\n", index);
}
}
}



}} 
