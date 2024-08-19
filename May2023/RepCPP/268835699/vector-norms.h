#ifndef __VECTOR_NORMS_MATH_CORE_H__
#define __VECTOR_NORMS_MATH_CORE_H__

#include <cmath>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"
#include "../macros/functions.h"
#include <math.h>

#include "../types/types.h"
#include "../meta/meta.h"

namespace __core__ {
namespace __math__ {
template <typename TRT=void,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__ __forceflatten__
RT __dot__(const Vector<V,N,AV>& v1,const Vector<U,N,AU>& v2) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v1,const Vector<U,N,AU>& v2),(result,v1,v2),
(result=__ma__<RM,V,U,RT>(v1(i),v2(i),result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v1,const Vector<U,N,AU>& v2),(result,v1,v2),
(result=__ma__<RM,V,U,RT>(v1(i),v2(i),result);))
#endif
return result;
}

template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>,enable_IT<(N>1)&&((!optimized)||(optimized&&(!is_fp_CE<V>())))> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<V,N,AV>& x) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& x),(result,x),(result=__ma__<RM>(x(i),x(i),result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& x),(result,x),(result=__ma__<RM>(x(i),x(i),result);))
#endif
return __sqrt__<FM,RM,RT>(result);
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<double,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(norm(N,x.x));
#else
RT result=0;
__unroll_cpu__(int,i,0,N,(RT&,const Vector<double,N,AV>&),(RT& result,const Vector<double,N,AV>& x),(result,x),(result=__ma__<RM,double,double,RT>(x(i),x(i),result);))
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<float,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(normf(N,x.x));
#else
RT result=0;
__unroll_cpu__(int,i,0,N,(RT&,const Vector<float,N,AV>&),(RT& result,const Vector<float,N,AV>& x),(result,x),(result=__ma__<RM,float,float,RT>(x(i),x(i),result);))
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<double,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(norm4d(x(0),x(1),x(2),x(3)));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
result=__ma__<RM>(x(3),x(3),result);
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<float,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(norm4df(x(0),x(1),x(2),x(3)));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
result=__ma__<RM>(x(3),x(3),result);
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<double,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(norm3d(x(0),x(1),x(2)));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<float,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(norm3df(x(0),x(1),x(2)));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<double,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(hypot(x(0),x(1)));
#else
RT result=__ma__<RM>(x(1),x(1),__mul__<RM>(x(0),x(0)));
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<float,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(hypotf(x(0),x(1)));
#else
RT result=__ma__<RM>(x(1),x(1),__mul__<RM>(x(0),x(0)));
return __sqrt__<FM,RM,RT>(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2__(const Vector<V,1,AV>& x) {
return __cast__<RT,RM>(__abs__(x(0)));
}

template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>,enable_IT<(N>1)&&((!optimized)||(optimized&&(!is_fp_CE<V>())))> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<V,N,AV>& x) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& x),(result,x),(result=__ma__<RM>(x(i),x(i),result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&),(RT& result,const Vector<V,N,AV>& x),(result,x),(result=__ma__<RM>(x(i),x(i),result);))
#endif
return result;
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<double,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(norm(N,x.x)));
#else
RT result=0;
__unroll_cpu__(int,i,0,N,(RT&,const Vector<double,N,AV>&),(RT& result,const Vector<double,N,AV>& x),(result,x),(result=__ma__<RM,double,double,RT>(x(i),x(i),result);))
return result;
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<float,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(normf(N,x.x)));
#else
RT result=0;
__unroll_cpu__(int,i,0,N,(RT&,const Vector<float,N,AV>&),(RT& result,const Vector<float,N,AV>& x),(result,x),(result=__ma__<RM,float,float,RT>(x(i),x(i),result);))
return result;
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<double,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(norm4d(x(0),x(1),x(2),x(3))));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
result=__ma__<RM>(x(3),x(3),result);
return result;
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<float,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(norm4df(x(0),x(1),x(2),x(3))));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
result=__ma__<RM>(x(3),x(3),result);
return result;
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<double,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(norm3d(x(0),x(1),x(2))));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
return result;
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<float,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(norm3df(x(0),x(1),x(2))));
#else
RT result=__mul__<RM>(x(0),x(0));
result=__ma__<RM>(x(1),x(1),result);
result=__ma__<RM>(x(2),x(2),result);
return result;
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<double,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(hypot(x(0),x(1))));
#else
RT result=__ma__<RM>(x(1),x(1),__mul__<RM>(x(0),x(0)));
return result;
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<float,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(__pow2__<RM>(hypotf(x(0),x(1))));
#else
RT result=__ma__<RM>(x(1),x(1),__mul__<RM>(x(0),x(0)));
return result;
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_squared__(const Vector<V,1,AV>& x) {
return __cast__<RT,RM>(__mul__<RM>(x(0),x(0)));
}

template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>,enable_IT<(N>1)&&((!optimized)||(optimized&&(!is_fp_CE<V>())))> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<V,N,AV>& x) {
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<double,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnorm(N,x.x));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,int N=0,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized&&(N>4)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<float,N,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnormf(N,x.x));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<double,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnorm4d(x(0),x(1),x(2),x(3)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<float,4,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnorm4df(x(0),x(1),x(2),x(3)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<double,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnorm3d(x(0),x(1),x(2)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<float,3,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rnorm3df(x(0),x(1),x(2)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),double,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<double,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rhypot(x(0),x(1)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),float,TRT>,enable_IT<optimized> = 0> __forceinline__  __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<float,2,AV>& x) {
#if defined(__CUDA_ARCH__)
return __cast__<RT,RM>(rhypotf(x(0),x(1)));
#else
RT result=__norm2_squared__<RT,optimized,FM,RM>(x);
return __rsqrt__(result);
#endif
}
template <typename TRT=float,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,uint AV=0,typename RT=conditional_T<is_same_CE<void,TRT>(),V,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2__(const Vector<V,1,AV>& x) {
return __cast__<RT,RM>(__rcp__<RM>(__abs__(x(0))));
}

template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>,enable_IT<(N>1)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#endif
return __sqrt__<FM,RM,RT>(result);
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_diff__(const Vector<V,1,AV>& v,const Vector<U,1,AU>& u) {
return __abs__(__sub__<RM>(v(0)-u(0)));
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>,enable_IT<(N>1)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_diff_squared__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#endif
return result;
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __norm2_diff_squared__(const Vector<V,1,AV>& v,const Vector<U,1,AU>& u) {
return __pow2__<RM>(__sub__<RM>(v(0)-u(0)));
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,int N=0,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>,enable_IT<(N>1)> = 0> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2_diff__(const Vector<V,N,AV>& v,const Vector<U,N,AU>& u) {
RT result=0;
#if defined(__CUDA_ARCH__)
#ifdef __PRAGMA_UNROLL__
#pragma unroll
#endif
__unroll_gpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#else
__unroll_cpu__(int,i,0,N,(RT&,const Vector<V,N,AV>&,const Vector<U,N,AU>&),(RT& result,const Vector<V,N,AV>& v,const Vector<U,N,AU>& u),(result,v,u),(
auto tmp=__sub__<RM>(v(i),u(i));
result=__ma__<RM>(tmp,tmp,result);))
#endif
return __rsqrt__(result);
}
template <typename TRT=double,bool optimized=true,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,uint AV=0,uint AU=0,typename RT=conditional_T<is_same_CE<void,TRT>(),higher_PT<V,U>,TRT>> __forceinline__ __optimize__ __host_device__  __forceflatten__
RT __rnorm2_diff__(const Vector<V,1,AV>& v,const Vector<U,1,AU>& u) {
return __rcp__<RM>(__abs__(__sub__<RM>(v(0)-u(0))));
}
}
}
#endif
