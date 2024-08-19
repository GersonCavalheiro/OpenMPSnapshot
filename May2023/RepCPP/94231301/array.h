
#pragma once

#include "platform.h"
#include "alloc.h"

namespace embree
{

template<typename T, size_t N>
class array_t
{
public:



__forceinline T* begin() const { return items; };
__forceinline T* end  () const { return items+N; };




__forceinline bool   empty    () const { return N == 0; }
__forceinline size_t size     () const { return N; }
__forceinline size_t max_size () const { return N; }




__forceinline       T& operator[](size_t i)       { assert(i < N); return items[i]; }
__forceinline const T& operator[](size_t i) const { assert(i < N); return items[i]; }

__forceinline       T& at(size_t i)       { assert(i < N); return items[i]; }
__forceinline const T& at(size_t i) const { assert(i < N); return items[i]; }

__forceinline T& front() const { assert(N > 0); return items[0]; };
__forceinline T& back () const { assert(N > 0); return items[N-1]; };

__forceinline       T* data()       { return items; };
__forceinline const T* data() const { return items; };

private:
T items[N];
};


template<typename T, size_t N>
class darray_t
{
public:

__forceinline darray_t () : M(0) {}

__forceinline darray_t (const T& v) : M(0) {
for (size_t i=0; i<N; i++) items[i] = v;
}



__forceinline T* begin() const { return items; };
__forceinline T* end  () const { return items+M; };




__forceinline bool   empty    () const { return M == 0; }
__forceinline size_t size     () const { return M; }
__forceinline size_t capacity () const { return N; }
__forceinline size_t max_size () const { return N; }

void resize(size_t new_size) {
assert(new_size < max_size());
M = new_size;
}



__forceinline void push_back(const T& v) 
{
assert(M+1 < max_size());
items[M++] = v;
}

__forceinline void pop_back() 
{
assert(!empty());
M--;
}

__forceinline void clear() {
M = 0;
}



__forceinline       T& operator[](size_t i)       { assert(i < M); return items[i]; }
__forceinline const T& operator[](size_t i) const { assert(i < M); return items[i]; }

__forceinline       T& at(size_t i)       { assert(i < M); return items[i]; }
__forceinline const T& at(size_t i) const { assert(i < M); return items[i]; }

__forceinline T& front() const { assert(M > 0); return items[0]; };
__forceinline T& back () const { assert(M > 0); return items[M-1]; };

__forceinline       T* data()       { return items; };
__forceinline const T* data() const { return items; };

private:
size_t M;
T items[N];
};


#define dynamic_large_stack_array(Ty,Name,N,max_stack_bytes) StackArray<Ty,max_stack_bytes> Name(N)
template<typename Ty, size_t max_stack_bytes>
struct __aligned(64) StackArray
{
__forceinline StackArray (const size_t N)
: N(N)
{
if (N*sizeof(Ty) <= max_stack_bytes) 
data = &arr[0];
else
data = (Ty*) alignedMalloc(N*sizeof(Ty),64); 
}

__forceinline ~StackArray () {
if (data != &arr[0]) alignedFree(data);
}

__forceinline operator       Ty* ()       { return data; }
__forceinline operator const Ty* () const { return data; }

__forceinline       Ty& operator[](const int i)       { assert(i>=0 && i<N); return data[i]; }
__forceinline const Ty& operator[](const int i) const { assert(i>=0 && i<N); return data[i]; }

__forceinline       Ty& operator[](const unsigned i)       { assert(i<N); return data[i]; }
__forceinline const Ty& operator[](const unsigned i) const { assert(i<N); return data[i]; }

#if defined(__X86_64__)
__forceinline       Ty& operator[](const size_t i)       { assert(i<N); return data[i]; }
__forceinline const Ty& operator[](const size_t i) const { assert(i<N); return data[i]; }
#endif

private:
Ty arr[max_stack_bytes/sizeof(Ty)];
Ty* data;
size_t N;
};
}
