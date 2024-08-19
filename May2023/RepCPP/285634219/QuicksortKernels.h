
#ifndef QUICKSORT_KNL_H
#define QUICKSORT_KNL_H

#include "Quicksort.h"

#pragma omp declare target
template <typename T>
void plus_prescan( T *a,  T *b) {
T av = *a;
T bv = *b;
*a = bv;
*b = bv + av;
}

template <typename T>
void bitonic_sort( T* sh_data, const uint localid)
{
for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
for (uint j = ulevel; j > 0; j >>= 1) {
uint pos = 2*localid - (localid & (j - 1));

uint direction = localid & ulevel;
uint av = sh_data[pos], bv = sh_data[pos + j];
const bool sortThem = av > bv;
const uint greater = Select(bv, av, sortThem);
const uint lesser  = Select(av, bv, sortThem);

sh_data[pos]     = Select(lesser, greater, direction);
sh_data[pos + j] = Select(greater, lesser, direction);
#pragma omp barrier
}
}

for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
uint pos = 2*localid - (localid & (j - 1));

uint av = sh_data[pos], bv = sh_data[pos + j];
const bool sortThem = av > bv;
sh_data[pos]      = Select(av, bv, sortThem);
sh_data[pos + j]  = Select(bv, av, sortThem);

#pragma omp barrier
}
}

template <typename T>
void sort_threshold( T* data_in, 
T* data_out,
uint start, 
uint end,  
T* temp, 
uint localid) 
{
uint tsum = end - start;
if (tsum == SORT_THRESHOLD) {
bitonic_sort(data_in+start, localid);
for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
data_out[start + i] = data_in[start + i];
}
} else if (tsum > 1) {
for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
if (i < tsum) {
temp[i] = data_in[start + i];
} else {
temp[i] = UINT_MAX;
}
}
#pragma omp barrier
bitonic_sort(temp, localid);

for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
data_out[start + i] = temp[i];
}
} else if (tsum == 1 && localid == 0) {
data_out[start] = data_in[start];
} 
}
#pragma omp end declare target

typedef struct workstack_record {
uint start;
uint end;
uint direction;
} workstack_record;

#define PUSH(START, END)       if (localid == 0) { \
++workstack_pointer; \
workstack_record wr = { (START), (END), direction ^ 1 }; \
workstack[workstack_pointer] = wr; \
} 

#endif
