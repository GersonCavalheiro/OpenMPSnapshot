

#ifndef ROCALUTION_HIP_HIP_UNORDERED_MAP_HPP_
#define ROCALUTION_HIP_HIP_UNORDERED_MAP_HPP_

#include "hip_atomics.hpp"

#include <hip/hip_runtime.h>
#include <limits>
#include <utility>

namespace rocalution
{
template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL = 31232527,
KeyType      EMPTY   = std::numeric_limits<KeyType>::max()>
class unordered_map
{
public:
__device__ __forceinline__ explicit unordered_map(KeyType* skeys, ValType* svals);
__device__ __forceinline__ ~unordered_map(void) {}

__device__ __forceinline__ void clear(void);
__device__ __forceinline__ constexpr KeyType empty_key(void) const
{
return EMPTY;
}

__device__ __forceinline__ bool insert(const KeyType& key,
const ValType& val = static_cast<ValType>(0));
__device__ __forceinline__ bool insert_or_add(const KeyType& key, const ValType& val);
__device__ __forceinline__ bool add(const KeyType& key, const ValType& val);
__device__ __forceinline__ bool contains(const KeyType& key) const;
__device__ __forceinline__ KeyType get_key(unsigned int i) const;
__device__ __forceinline__ ValType get_val(unsigned int i) const;
__device__ __forceinline__ std::pair<KeyType, ValType> get_pair(unsigned int i) const;

__device__ __forceinline__ void store_sorted(KeyType* keys, ValType* vals) const;
__device__ __forceinline__ void store_sorted_with_perm(const int* perm,
ValType    alpha,
KeyType*   keys,
ValType*   vals) const;

__device__ __forceinline__ void sort(void);

private:
KeyType* keys_;
ValType* vals_;
};

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::unordered_map(
KeyType* skeys, ValType* svals)
{
this->keys_ = skeys;
this->vals_ = svals;

this->clear();
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ void
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::clear(void)
{
unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
for(unsigned int i = tid; i < SIZE; i += NTHREADS)
{
this->keys_[i] = EMPTY;
this->vals_[i] = static_cast<ValType>(0);
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
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert(const KeyType& key,
const ValType& val)
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
this->vals_[hash] = val;

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
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::insert_or_add(
const KeyType& key, const ValType& val)
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
atomicAdd(&this->vals_[hash], val);

return false;
}
else if(this->keys_[hash] == EMPTY)
{
if(atomicCAS(&this->keys_[hash], EMPTY, key) == EMPTY)
{
atomicAdd(&this->vals_[hash], val);

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
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::add(const KeyType& key,
const ValType& val)
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
atomicAdd(&this->vals_[hash], val);

return true;
}
else if(this->keys_[hash] == EMPTY)
{
return false;
}
else
{
hash = (hash + hash_inc) & (SIZE - 1);
}
}

return false;
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ bool
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::contains(
const KeyType& key) const
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
return true;
}
else if(this->keys_[hash] == EMPTY)
{
return false;
}
else
{
hash = (hash + hash_inc) & (SIZE - 1);
}
}

return false;
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ KeyType
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_key(
unsigned int i) const
{
return (i >= 0 && i < SIZE) ? this->keys_[i] : EMPTY;
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ ValType
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_val(
unsigned int i) const
{
return (i >= 0 && i < SIZE) ? this->vals_[i] : static_cast<ValType>(0);
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ std::pair<KeyType, ValType>
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::get_pair(
unsigned int i) const
{
return ((i >= 0 && i < SIZE) ? std::make_pair(this->keys_[i], this->vals_[i])
: std::make_pair(EMPTY, static_cast<ValType>(0)));
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ void
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::store_sorted(
KeyType* keys, ValType* vals) const
{
unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
for(unsigned int i = tid; i < SIZE; i += NTHREADS)
{
KeyType key = this->keys_[i];

if(key == EMPTY)
{
continue;
}

int idx = 0;

unsigned int cnt = 0;

while(cnt < SIZE)
{
if(key > this->keys_[cnt])
{
++idx;
}

++cnt;
}

keys[idx] = key;
vals[idx] = this->vals_[i];
}
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ void
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::store_sorted_with_perm(
const int* perm, ValType alpha, KeyType* keys, ValType* vals) const
{
unsigned int tid = threadIdx.x & (NTHREADS - 1);

#pragma unroll 4
for(unsigned int i = tid; i < SIZE; i += NTHREADS)
{
KeyType key = this->keys_[i];

if(key == EMPTY)
{
continue;
}

int idx = 0;

unsigned int cnt = 0;

while(cnt < SIZE)
{
if(key > this->keys_[cnt])
{
++idx;
}

++cnt;
}

keys[idx] = perm[key];
vals[idx] = alpha * this->vals_[i];
}
}

template <typename KeyType,
typename ValType,
unsigned int SIZE,
unsigned int NTHREADS,
unsigned int HASHVAL,
KeyType      EMPTY>
__device__ __forceinline__ void
unordered_map<KeyType, ValType, SIZE, NTHREADS, HASHVAL, EMPTY>::sort(void)
{
if(NTHREADS < warpSize)
{
__threadfence_block();
}
else
{
__syncthreads();
}

unsigned int tid = threadIdx.x & (NTHREADS - 1);

KeyType      keys[SIZE / NTHREADS];
ValType      vals[SIZE / NTHREADS];
unsigned int idx[SIZE / NTHREADS];

for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
{
keys[i] = this->keys_[tid + NTHREADS * i];
vals[i] = this->vals_[tid + NTHREADS * i];
}

#pragma unroll 4
for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
{
if(keys[i] == EMPTY)
{
continue;
}

idx[i] = 0;

unsigned int cnt = 0;

while(cnt < SIZE)
{
if(keys[i] > this->keys_[cnt])
{
++idx[i];
}

++cnt;
}
}

for(unsigned int i = tid; i < SIZE; i += NTHREADS)
{
this->keys_[i] = EMPTY;
}

for(unsigned int i = 0; i < SIZE / NTHREADS; ++i)
{
if(keys[i] == EMPTY)
{
continue;
}

this->keys_[idx[i]] = keys[i];
this->vals_[idx[i]] = vals[i];
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

} 

#endif 
