
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/core/alignment.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <cassert>


HYDRA_THRUST_BEGIN_NS

namespace cuda_cub {
namespace launcher {

struct triple_chevron
{
typedef size_t Size;
dim3 const grid;
dim3 const block;
Size const shared_mem;
cudaStream_t const stream;

HYDRA_THRUST_RUNTIME_FUNCTION
triple_chevron(dim3         grid_,
dim3         block_,
Size         shared_mem_ = 0,
cudaStream_t stream_     = 0)
: grid(grid_),
block(block_),
shared_mem(shared_mem_),
stream(stream_) {}

#if 0
template<class K, class... Args>
cudaError_t __host__
doit_host(K k, Args const&... args) const
{
k<<<grid, block, shared_mem, stream>>>(args...);
return cudaPeekAtLastError();
}
#else
template <class K, class _0>
cudaError_t __host__
doit_host(K k, _0 x0) const
{
k<<<grid, block, shared_mem, stream>>>(x0);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE);
return cudaPeekAtLastError();
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE, class _xF>
cudaError_t __host__
doit_host(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE, _xF xF) const
{
k<<<grid, block, shared_mem, stream>>>(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE,xF);
return cudaPeekAtLastError();
}
#endif

template<class T>
size_t __device__
align_up(size_t offset) const
{
size_t alignment = alignment_of<T>::value;
return alignment * ((offset + (alignment - 1))/ alignment);
}

#if 0
size_t __device__ argument_pack_size(size_t size) const { return size; }
template <class Arg, class... Args>
size_t __device__
argument_pack_size(size_t size, Arg const& arg, Args const&... args) const
{
size = align_up<Arg>(size);
return argument_pack_size(size + sizeof(Arg), args...);
}
#else
template <class Arg>
size_t __device__
argument_pack_size(size_t size, Arg) const
{
return align_up<Arg>(size) + sizeof(Arg);
}
template <class Arg, class _0>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0);
}
template <class Arg, class _0, class _1>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1);
}
template <class Arg, class _0, class _1, class _2>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2);
}
template <class Arg, class _0, class _1, class _2, class _3>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE, class _xF>
size_t __device__
argument_pack_size(size_t size, Arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE, _xF xF) const
{
return argument_pack_size(align_up<Arg>(size) + sizeof(Arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, xF);
}
#endif 

template <class Arg>
size_t __device__ copy_arg(char* buffer, size_t offset, Arg arg) const
{
offset = align_up<Arg>(offset);
for (int i = 0; i != sizeof(Arg); ++i)
buffer[offset+i] = *((char*)&arg + i);
return offset + sizeof(Arg);
}

#if 0
void __device__ fill_arguments(char*, size_t) const {}
template<class Arg, class... Args>
void __device__
fill_arguments(char* buffer, size_t offset, Arg const& arg, Args const& ... args) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), args...);
}
#else
template<class Arg>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg) const
{
copy_arg(buffer, offset, arg);
}
template<class Arg, class _0>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0);
}
template <class Arg, class _0, class _1>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1);
}
template <class Arg, class _0, class _1, class _2>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2);
}
template <class Arg, class _0, class _1, class _2, class _3>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
}
template <class Arg, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE, class _xF>
void __device__
fill_arguments(char* buffer, size_t offset, Arg arg, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE, _xF xF) const
{
fill_arguments(buffer, copy_arg(buffer, offset, arg), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, xF);
}
#endif 

#if 0
template<class K, class... Args>
cudaError_t __device__
doit_device(K k, Args const&... args) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,args...);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, args...);
status = launch_device(k, param_buffer);
#endif
return status;
}
#else
template<class K, class _0>
cudaError_t __device__
doit_device(K k, _0 x0) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
#endif
return status;
}
template <class K, class _0, class _1>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
#endif
return status;
}
template <class K, class _0, class _1, class _2>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
HYDRA_THRUST_UNUSED_VAR(xB);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
HYDRA_THRUST_UNUSED_VAR(xB);
HYDRA_THRUST_UNUSED_VAR(xC);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
HYDRA_THRUST_UNUSED_VAR(xB);
HYDRA_THRUST_UNUSED_VAR(xC);
HYDRA_THRUST_UNUSED_VAR(xD);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
HYDRA_THRUST_UNUSED_VAR(xB);
HYDRA_THRUST_UNUSED_VAR(xC);
HYDRA_THRUST_UNUSED_VAR(xD);
HYDRA_THRUST_UNUSED_VAR(xE);
#endif
return status;
}
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE, class _xF>
cudaError_t __device__
doit_device(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC,_xD xD, _xE xE, _xF xF) const
{
cudaError_t status = cudaErrorNotSupported;
#if __HYDRA_THRUST_HAS_CUDART__
const size_t size = argument_pack_size(0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE,xF);
void *param_buffer = cudaGetParameterBuffer(64,size);
fill_arguments((char*)param_buffer, 0, x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,xA,xB,xC,xD,xE,xF);
status = launch_device(k, param_buffer);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(x0);
HYDRA_THRUST_UNUSED_VAR(x1);
HYDRA_THRUST_UNUSED_VAR(x2);
HYDRA_THRUST_UNUSED_VAR(x3);
HYDRA_THRUST_UNUSED_VAR(x4);
HYDRA_THRUST_UNUSED_VAR(x5);
HYDRA_THRUST_UNUSED_VAR(x6);
HYDRA_THRUST_UNUSED_VAR(x7);
HYDRA_THRUST_UNUSED_VAR(x8);
HYDRA_THRUST_UNUSED_VAR(x9);
HYDRA_THRUST_UNUSED_VAR(xA);
HYDRA_THRUST_UNUSED_VAR(xB);
HYDRA_THRUST_UNUSED_VAR(xC);
HYDRA_THRUST_UNUSED_VAR(xD);
HYDRA_THRUST_UNUSED_VAR(xE);
HYDRA_THRUST_UNUSED_VAR(xF);
#endif
return status;
}
#endif 

template <class K>
cudaError_t __device__
launch_device(K k, void* buffer) const
{
#if __HYDRA_THRUST_HAS_CUDART__
return cudaLaunchDevice((void*)k,
buffer,
dim3(grid),
dim3(block),
shared_mem,
stream);
#else
HYDRA_THRUST_UNUSED_VAR(k);
HYDRA_THRUST_UNUSED_VAR(buffer);
return cudaErrorNotSupported;
#endif
}


#ifdef __CUDA_ARCH__
#define HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE doit_device
#else
#define HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE doit_host
#endif

#if 0
__hydra_thrust_exec_check_disable__
template <class K, class... Args>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, Args const&... args) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, args...);
}
#else
__hydra_thrust_exec_check_disable__
template <class K, class _0>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE);
}
__hydra_thrust_exec_check_disable__
template <class K, class _0, class _1, class _2, class _3, class _4, class _5, class _6, class _7, class _8, class _9, class _xA, class _xB, class _xC, class _xD, class _xE, class _xF>
cudaError_t HYDRA_THRUST_FUNCTION
doit(K k, _0 x0, _1 x1, _2 x2, _3 x3, _4 x4, _5 x5, _6 x6, _7 x7, _8 x8, _9 x9, _xA xA, _xB xB, _xC xC, _xD xD, _xE xE, _xF xF) const
{
return HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE(k, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, xF);
}
#endif
#undef HYDRA_THRUST_TRIPLE_LAUNCHER_HOSTDEVICE
}; 

}    
}    

HYDRA_THRUST_END_NS
