#ifndef __FOR_SEQUENCE_META_CUH__
#define __FOR_SEQUENCE_META_CUH__

#include <type_traits>

#include <cuda.h>

#include "constexpr.h"
#include "types.h"

namespace __core__ {
namespace __meta__ {
template<typename FT,typename CA> struct const_for_sequence {
static constexpr int size=CA::size;
static_assert((size>0),"The index sequence doesn't have values to iterate!");
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<I==(N-1)> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
typedef typename CA::nth_VT<I> v_T;
v_T val=c_T::value;
FT::fn(val,args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I<(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
typedef typename CA::nth_VT<I> v_T;
v_T val=c_T::value;
FT::fn(val,args...);
__iterator__<I+1,N,Args...>(args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I==(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __const_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
FT::template fn<c_T>(args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I<(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __const_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
FT::template fn<c_T>(args...);
__const_iterator__<I+1,N,Args...>(args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<I==(N-1)> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __indexed_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
typedef typename CA::nth_VT<I> v_T;
v_T val=c_T::value;
FT::fn(I,val,args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I<(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __indexed_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
typedef typename CA::nth_VT<I> v_T;
v_T val=c_T::value;
FT::fn(I,val,args...);
__indexed_iterator__<I+1,N,Args...>(args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I==(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __const_indexed_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
FT::template fn<I,c_T>(args...);
}
#pragma hd_warning_disable
template <int I,int N,typename... Args,enable_IT<(I<(N-1))> = 0> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void __const_indexed_iterator__(Args... args) {
typedef typename CA::nth_T<I> c_T;
FT::template fn<I,c_T>(args...);
__const_indexed_iterator__<I+1,N,Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
__iterator__<0,size,Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static  __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
__const_iterator__<0,size,Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator_margs(Args... args) {
__indexed_iterator__<0,size,Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static  __host__ __device__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator_margs(Args... args) {
__const_indexed_iterator__<0,size,Args...>(args...);
}
};
}
}
#endif
