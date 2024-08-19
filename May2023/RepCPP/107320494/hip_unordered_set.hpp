

#ifndef ROCALUTION_HIP_HIP_UNORDERED_SET_HPP_
#define ROCALUTION_HIP_HIP_UNORDERED_SET_HPP_

#include "hip_atomics.hpp"

#include <hip/hip_runtime.h>
#include <limits>

namespace rocalution
{
template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL = 31232527,
KeyType      EMPTY   = std::numeric_limits<KeyType>::max()>
class unordered_set
{
public:
__device__ __forceinline__ explicit unordered_set(KeyType* skeys);
__device__ __forceinline__ ~unordered_set(void) {}

__device__ __forceinline__ void clear(void);
__device__ __forceinline__ constexpr KeyType empty_key(void) const
{
return EMPTY;
}

__device__ __forceinline__ bool insert(const KeyType& key);
__device__ __forceinline__ bool contains(const KeyType& key) const;
__device__ __forceinline__ KeyType get_key(unsigned int i) const;

private:
KeyType* keys_;
};

template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__
unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::unordered_set(KeyType* skeys)
{
this->keys_ = skeys;

this->clear();
}

template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ void
unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::clear(void)
{
unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
for(unsigned int i = tid; i < SIZE; i += NTHREADS)
{
this->keys_[i] = EMPTY;
}

if(NTHREADS < warpSize)
{
__threadfence_block();
}
else
{
__syncthreads();
}
}

template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert(const KeyType& key)
{
if(key == EMPTY)
{
return false;
}

unsigned int hash     = (key * HASHVAL) & (SIZE - 1);
unsigned int hash_inc = 1 + 2 * (key & (NTHREADS - 1));

while(true)
{
if(this->keys_[hash] == key)
{
return false;
}
else if(this->keys_[hash] == EMPTY)
{
if(atomicCAS(&this->keys_[hash], EMPTY, key) == EMPTY)
{
return true;
}
}
else
{
hash = (hash + hash_inc) & (SIZE - 1);
}
}

return false;
}

template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::contains(const KeyType& key) const
{
if(key == EMPTY)
{
return false;
}

unsigned int hash     = (key * HASHVAL) & (SIZE - 1);
unsigned int hash_inc = 1 + 2 * (key & (NTHREADS - 1));

while(true)
{
if(this->keys_[hash] == EMPTY)
{
return false;
}
else if(this->keys_[hash] == key)
{
return true;
}
else
{
hash = (hash + hash_inc) & (SIZE - 1);
}
}

return false;
}

template <typename KeyType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ KeyType
unordered_set<KeyType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_key(unsigned int i) const
{
return (i >= 0 && i < SIZE) ? this->keys_[i] : EMPTY;
}

} 

#endif 
