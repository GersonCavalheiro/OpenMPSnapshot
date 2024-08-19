#ifndef __VECTOR_ELEMENTAL_FUNCTIONS_MATH_CORE_H__
#define __VECTOR_ELEMENTAL_FUNCTIONS_MATH_CORE_H__

#include "../macros/macros.h"
#include "../meta/meta.h"
#include "../types/types.h"
#include "arithmetic.h"
#include "elemental-functions.h"
#include "vector-arithmetic.h"

namespace __core__ {
namespace __math__ {
template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __floor__(const Vector<V,N,AV>& v) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__floor__(v(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__floor__(v(i));))
#endif
return result;
}
template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __ceil__(const Vector<V,N,AV>& v) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__ceil__(v(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__ceil__(v(i));))
#endif
return result;
}

template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __abs__(const Vector<V,N,AV>& v) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__abs__(v(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__abs__(v(i));))
#endif
return result;
}
template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __sign__(const Vector<V,N,AV>& v) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__sign__(v(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__sign__(v(i));))
#endif
return result;
}

template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,RVT>(),V,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __max__(const Vector<V,N,AV>& v) {
RT result=v(0);
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result=__max__(v(i),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result=__max__(v(i),result);))
#endif
return result;
}
template <typename RVT=void,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __max__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__max__(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__max__(v(i),x);))
#endif
return result;
}
template <typename RVT=void,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __max__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__max__(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__max__(v(i),u(i));))
#endif
return result;
}
template <typename RVT=void,typename V=void,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,RVT>(),V,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __min__(const Vector<V,N,AV>& v) {
RT result=v(0);
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result=__min__(v(i),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result=__min__(v(i),result);))
#endif
return result;
}
template <typename RVT=void,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __min__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__min__(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__min__(v(i),x);))
#endif
return result;
}
template <typename RVT=void,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __min__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__min__(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__min__(v(i),u(i));))
#endif
return result;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,RVT>(),higher_PT<V,U>,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __max_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=__sub__<RM>(v(0),u(0));
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__max__(__sub__<RM>(v(i),u(i)),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__max__(__sub__<RM>(v(i),u(i)),result);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,RVT>(),higher_PT<V,U>,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __min_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=__sub__<RM>(v(0),u(0));
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__min__(__sub__<RM>(v(i),u(i)),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__min__(__sub__<RM>(v(i),u(i)),result);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,RVT>(),higher_PT<V,U>,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __max_abs_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=__sub__<RM>(v(0),u(0));
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__max__(__abs__(__sub__<RM>(v(i),u(i))),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__max__(__abs__(__sub__<RM>(v(i),u(i))),result);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,RVT>(),higher_PT<V,U>,RVT>,enable_IT<!is_vector_CE<RVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __min_abs_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=__sub__<RM>(v(0),u(0));
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__min__(__abs__(__sub__<RM>(v(i),u(i))),result);))
#else
__unroll_cpu__(int,i,1,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),
(result=__min__(__abs__(__sub__<RM>(v(i),u(i))),result);))
#endif
return result;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_add__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__abs__(__add__<RM,V,U>(v(i),x));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__abs__(__add__<RM,V,U>(v(i),x));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__abs__(__add__<RM,V,U>(v(i),u(i)));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__abs__(__add__<RM,V,U>(v(i),u(i)));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename X=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha),(result,v,u,alpha),
(result[i]=__abs__(__ma__<RM>(v(i),alpha,u(i)));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha),(result,v,u,alpha),
(result[i]=__abs__(__ma__<RM>(v(i),alpha,u(i)));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename X=void,typename Y=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X,const Y),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta),(result,v,u,alpha,beta),
(result[i]=__abs__(__ma__<RM>(v(i),alpha,__mul__<RM>(u(i),beta)));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X,const Y),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta),(result,v,u,alpha,beta),
(result[i]=__abs__(__ma__<RM>(v(i),alpha,__mul__<RM>(u(i),beta)));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_sub__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__abs__(__sub__<RM,V,U>(v(i),x));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__abs__(__sub__<RM,V,U>(v(i),x));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __abs_sub__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__abs__(__sub__<RM,V,U>(v(i),u(i)));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__abs__(__sub__<RM,V,U>(v(i),u(i)));))
#endif
return result;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,3,AV>,Vector<U,3,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __cross__(const Vector<V,3,AV>& v,const Vector<U,3,AU>& u) {
RT result;
result[0]=__sub__<RM>(__mul__<RM>(v(1),u(2)),__mul__<RM>(v(2),u(1)));
result[1]=__sub__<RM>(__mul__<RM>(v(2),u(0)),__mul__<RM>(v(0),u(2)));
result[2]=__sub__<RM>(__mul__<RM>(v(0),u(1)),__mul__<RM>(v(1),u(0)));
return result;
}

template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __angle__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
return __acos__(__dot__<TRT,RM>(v,u)/(__mul__<RM>(__norm2__<TRT,optimized,FM,RM>(v),__norm2__<TRT,optimized,FM,RM>(u))));
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __angle_cos__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
return __dot__<TRT,RM>(v,u)/(__mul__<RM>(__norm2__<TRT,optimized,FM,RM>(v),__norm2__<TRT,optimized,FM,RM>(u)));
}

template <typename RVT,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RET=basal_T<RVT>,enable_IT<same_dimensions<RVT,Vector<V,N,AV>>()&&(!is_same_CE<RVT,Vector<V,N,AV>>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
RVT __cast__(const Vector<V,N,AV>& v) {
RVT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RVT&,const Vector<V,N,AV>&),(RVT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__cast__<RET,RM>(v(i));))
#else
__unroll_cpu__(int,i,0,N,(RVT&,const Vector<V,N,AV>&),(RVT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__cast__<RET,RM>(v(i));))
#endif
return result;
}
template <typename RVT,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RET=basal_T<RVT>,enable_IT<is_same_CE<RVT,Vector<V,N,AV>>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RVT __cast__(const Vector<V,N,AV>& v) {
return v;
}
}
}
#endif
