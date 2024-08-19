#ifndef __VECTOR_ARITHMETIC_MATH_CORE_H__
#define __VECTOR_ARITHMETIC_MATH_CORE_H__

#include "../macros/macros.h"
#include "../meta/meta.h"
#include "../types/types.h"
#include "arithmetic.h"

namespace __core__ {
namespace __math__ {
namespace __vprivate_arithmetic__ {
template <typename...T> struct ReturnTypeH;
template <typename RT,typename VT> struct ReturnTypeH<RT,VT> {
using type=conditional_T<is_same_CE<void,RT>(),VT,RT>;
};
template <typename RT,typename VT,typename U> struct ReturnTypeH<RT,VT,U> {
using VET=underlying_T<VT>;
using UET=underlying_T<U>;
using HT=higher_PT<VET,UET>;
static constexpr int dim=vdim<VT>();
static constexpr uint alignment=valignment<VT>()==valignment<U>()?valignment<VT>():__type_alignment__::valignment<HT>();
using VTA=Vector<HT,dim,alignment>;
using type=conditional_T<is_same_CE<void,RT>(),VTA,RT>;
};
template <typename RT,typename VT,typename U,typename W> struct ReturnTypeH<RT,VT,U,W> {
using VET=underlying_T<VT>;
using UET=underlying_T<U>;
using WET=underlying_T<W>;
using HT=higher_PT<VET,higher_PT<UET,WET>>;
static constexpr int dim=vdim<VT>();
static constexpr uint alignment=((valignment<VT>()==valignment<U>())&&(valignment<VT>()==valignment<W>()))?valignment<VT>():__type_alignment__::valignment<HT>();
using VTA=Vector<HT,dim,alignment>;
using type=conditional_T<is_same_CE<void,RT>(),VTA,RT>;
};
template <typename...T> using ReturnType=typename ReturnTypeH<T...>::type;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __add__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__add__<RM,V,U>(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__add__<RM,V,U>(v(i),x);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__add__<RM,V,U>(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__add__<RM,V,U>(v(i),u(i));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename X=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha),(result,v,u,alpha),
(result[i]=__ma__<RM>(v(i),alpha,u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha),(result,v,u,alpha),
(result[i]=__ma__<RM>(v(i),alpha,u(i));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename X=void,typename Y=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __add__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X,const Y),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta),(result,v,u,alpha,beta),
(result[i]=__ma__<RM>(v(i),alpha,__mul__<RM>(u(i),beta));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const X,const Y),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const X alpha,const Y beta),(result,v,u,alpha,beta),
(result[i]=__ma__<RM>(v(i),alpha,__mul__<RM>(u(i),beta));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __sub__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__sub__<RM,V,U>(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__sub__<RM,V,U>(v(i),x);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __sub__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__sub__<RM,V,U>(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__sub__<RM,V,U>(v(i),u(i));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __mul__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__mul__<RM,V,U>(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__mul__<RM,V,U>(v(i),x);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __mul__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__mul__<RM,V,U>(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__mul__<RM,V,U>(v(i),u(i));))
#endif
return result;
}
template <typename RVT=void,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __div__(const Vector<V,N,AV>& v,const U x) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__div__<FM,RM,V,U>(v(i),x);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U),(RT& result,const Vector<V,N,AV>& v,const U x),(result,v,x),(result[i]=__div__<FM,RM,V,U>(v(i),x);))
#endif
return result;
}
template <typename RVT=void,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __div__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__div__<FM,RM,V,U>(v(i),u(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(result[i]=__div__<FM,RM,V,U>(v(i),u(i));))
#endif
return result;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename W=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<(!is_vector_CE<U>())&&(!is_vector_CE<W>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __ma__(const Vector<V,N,AV>& v,const U u,const W w) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U,const W),(RT& result,const Vector<V,N,AV>& v,const U u,const W w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u,w);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U,const W),(RT& result,const Vector<V,N,AV>& v,const U u,const W w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u,w);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename W=void,int N=0,uint AV=0,uint AW=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,U>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __ma__(const Vector<V,N,AV>& v,const U u,const Vector<W,N,AW> w) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U,const Vector<W,N,AW>),(RT& result,const Vector<V,N,AV>& v,const U u,const Vector<W,N,AW> w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u,w(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const U,const Vector<W,N,AW>),(RT& result,const Vector<V,N,AV>& v,const U u,const Vector<W,N,AW> w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u,w(i));))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename W=void,int N=0,uint AV=0,uint AU=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>,enable_IT<!is_vector_CE<W>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
RT __ma__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const W w) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const W),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const W w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u(i),w);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const W),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const W w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u(i),w);))
#endif
return result;
}
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,typename W=void,int N=0,uint AV=0,uint AU=0,uint AW=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>,Vector<U,N,AU>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __ma__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const Vector<W,N,AW> w) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const Vector<W,N,AW>),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const Vector<W,N,AW> w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u(i),w(i));))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&,const Vector<W,N,AW>),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u,const Vector<W,N,AW> w),(result,v,u,w),
(result[i]=__ma__<RM>(v(i),u(i),w(i));))
#endif
return result;
}

template <typename RVT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RT=__vprivate_arithmetic__::ReturnType<RVT,Vector<V,N,AV>>,typename RET=basal_T<RT>> static inline __forceflatten__ __optimize__ __host_device__
RT __rcp__(const Vector<V,N,AV>& v) {
RT result;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__rcp__<RM>(v(i));))
#else
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& v),(result,v),(result[i]=__rcp__<RM>(v(i));))
#endif
return result;
}
}
}
#endif
