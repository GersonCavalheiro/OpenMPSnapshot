#ifndef __FOR_META_CUH__
#define __FOR_META_CUH__

#include <type_traits>
#include <utility>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

namespace __core__ {
namespace __meta__ {
template<typename FT,typename int_T,int_T I,int_T E> struct const_for {
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA>=E)>::type iterator(Args... args)	{
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA>=E)>::type iterator(FT& fo,Args... args) {
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA>=E)>::type const_iterator(Args... args) {
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA>=E)>::type const_iterator(FT& fo,Args... args) {
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA<E)>::type iterator(Args... args)	{
FT::fn(I,args...);
const_for<FT,int_T,I+1,E>::template iterator<Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA<E)>::type iterator(FT& fo,Args... args) {
fo(I,args...);
const_for<FT,int_T,I+1,E>::template iterator<Args...>(fo,args...);
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA<E)>::type const_iterator(Args... args) {
FT::template fn<I>(args...);
const_for<FT,int_T,I+1,E>::template const_iterator<Args...>(args...);
}
#pragma hd_warning_disable
template <typename... Args,int_T IA=I> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
typename std::enable_if<(IA<E)>::type const_iterator(FT& fo,Args... args) {
fo.template operator()<I>(args...);
const_for<FT,int_T,I+1,E>::template const_iterator<Args...>(fo,args...);
}
};
template<typename FT,typename int_T,int_T E> struct const_for<FT,int_T,E,E> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__  inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args)	{
FT::fn(E,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(E,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<E>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<E>(args...);
}
};
template<typename FT> struct const_for<FT,int,0,1> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0,args...);
FT::fn(1,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0,args...);
fo(1,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0>(args...);
FT::template fn<1>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0>(args...);
fo.template operator()<1>(args...);
}
};
template<typename FT> struct const_for<FT,int,0,2> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0,args...);
FT::fn(1,args...);
FT::fn(2,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0,args...);
fo(1,args...);
fo(2,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0>(args...);
FT::template fn<1>(args...);
FT::template fn<2>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0>(args...);
fo.template operator()<1>(args...);
fo.template operator()<2>(args...);
}
};
template<typename FT> struct const_for<FT,int,0,3> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0,args...);
FT::fn(1,args...);
FT::fn(2,args...);
FT::fn(3,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0,args...);
fo(1,args...);
fo(2,args...);
fo(3,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0>(args...);
FT::template fn<1>(args...);
FT::template fn<2>(args...);
FT::template fn<3>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0>(args...);
fo.template operator()<1>(args...);
fo.template operator()<2>(args...);
fo.template operator()<3>(args...);
}
};
template<typename FT> struct const_for<FT,int,0,4> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0,args...);
FT::fn(1,args...);
FT::fn(2,args...);
FT::fn(3,args...);
FT::fn(4,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0,args...);
fo(1,args...);
fo(2,args...);
fo(3,args...);
fo(4,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0>(args...);
FT::template fn<1>(args...);
FT::template fn<2>(args...);
FT::template fn<3>(args...);
FT::template fn<4>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0>(args...);
fo.template operator()<1>(args...);
fo.template operator()<2>(args...);
fo.template operator()<3>(args...);
fo.template operator()<4>(args...);
}
};

template<typename FT> struct const_for<FT,unsigned int,0U,1U> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0U,args...);
FT::fn(1U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0U,args...);
fo(1U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0U>(args...);
FT::template fn<1U>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0U>(args...);
fo.template operator()<1U>(args...);
}
};
template<typename FT> struct const_for<FT,unsigned int,0U,2U> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0U,args...);
FT::fn(1U,args...);
FT::fn(2U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0U,args...);
fo(1U,args...);
fo(2U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0U>(args...);
FT::template fn<1U>(args...);
FT::template fn<2U>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0U>(args...);
fo.template operator()<1U>(args...);
fo.template operator()<2U>(args...);
}
};
template<typename FT> struct const_for<FT,unsigned int,0U,3U> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0U,args...);
FT::fn(1U,args...);
FT::fn(2U,args...);
FT::fn(3U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0U,args...);
fo(1U,args...);
fo(2U,args...);
fo(3U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0U>(args...);
FT::template fn<1U>(args...);
FT::template fn<2U>(args...);
FT::template fn<3U>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0U>(args...);
fo.template operator()<1U>(args...);
fo.template operator()<2U>(args...);
fo.template operator()<3U>(args...);
}
};
template<typename FT> struct const_for<FT,unsigned int,0U,4U> {
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(Args... args) {
FT::fn(0U,args...);
FT::fn(1U,args...);
FT::fn(2U,args...);
FT::fn(3U,args...);
FT::fn(4U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void iterator(FT& fo,Args... args) {
fo(0U,args...);
fo(1U,args...);
fo(2U,args...);
fo(3U,args...);
fo(4U,args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(Args... args) {
FT::template fn<0U>(args...);
FT::template fn<1U>(args...);
FT::template fn<2U>(args...);
FT::template fn<3U>(args...);
FT::template fn<4U>(args...);
}
#pragma hd_warning_disable
template <typename... Args> static __HOST__ __DEVICE__ inline __attribute__((always_inline)) __attribute__((flatten)) __attribute__((optimize(3)))
void const_iterator(FT& fo,Args... args) {
fo.template operator()<0U>(args...);
fo.template operator()<1U>(args...);
fo.template operator()<2U>(args...);
fo.template operator()<3U>(args...);
fo.template operator()<4U>(args...);
}
};
}
}
#endif
