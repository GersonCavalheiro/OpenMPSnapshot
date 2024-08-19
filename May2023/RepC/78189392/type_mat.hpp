#pragma once
#include "precision.hpp"
namespace glm{
namespace detail
{
template <typename T, precision P, template <class, precision> class colType, template <class, precision> class rowType>
struct outerProduct_trait{};
}
template <typename T, precision P> struct tvec2;
template <typename T, precision P> struct tvec3;
template <typename T, precision P> struct tvec4;
template <typename T, precision P> struct tmat2x2;
template <typename T, precision P> struct tmat2x3;
template <typename T, precision P> struct tmat2x4;
template <typename T, precision P> struct tmat3x2;
template <typename T, precision P> struct tmat3x3;
template <typename T, precision P> struct tmat3x4;
template <typename T, precision P> struct tmat4x2;
template <typename T, precision P> struct tmat4x3;
template <typename T, precision P> struct tmat4x4;
typedef tmat2x2<float, lowp>		lowp_mat2;
typedef tmat2x2<float, mediump>		mediump_mat2;
typedef tmat2x2<float, highp>		highp_mat2;
typedef tmat2x2<float, lowp>		lowp_mat2x2;
typedef tmat2x2<float, mediump>		mediump_mat2x2;
typedef tmat2x2<float, highp>		highp_mat2x2;
typedef tmat2x3<float, lowp>		lowp_mat2x3;
typedef tmat2x3<float, mediump>		mediump_mat2x3;
typedef tmat2x3<float, highp>		highp_mat2x3;
typedef tmat2x4<float, lowp>		lowp_mat2x4;
typedef tmat2x4<float, mediump>		mediump_mat2x4;
typedef tmat2x4<float, highp>		highp_mat2x4;
typedef tmat3x2<float, lowp>		lowp_mat3x2;
typedef tmat3x2<float, mediump>		mediump_mat3x2;
typedef tmat3x2<float, highp>		highp_mat3x2;
typedef tmat3x3<float, lowp>		lowp_mat3;
typedef tmat3x3<float, mediump>		mediump_mat3;
typedef tmat3x3<float, highp>		highp_mat3;
typedef tmat3x3<float, lowp>		lowp_mat3x3;
typedef tmat3x3<float, mediump>		mediump_mat3x3;
typedef tmat3x3<float, highp>		highp_mat3x3;
typedef tmat3x4<float, lowp>		lowp_mat3x4;
typedef tmat3x4<float, mediump>		mediump_mat3x4;
typedef tmat3x4<float, highp>		highp_mat3x4;
typedef tmat4x2<float, lowp>		lowp_mat4x2;
typedef tmat4x2<float, mediump>		mediump_mat4x2;
typedef tmat4x2<float, highp>		highp_mat4x2;
typedef tmat4x3<float, lowp>		lowp_mat4x3;
typedef tmat4x3<float, mediump>		mediump_mat4x3;
typedef tmat4x3<float, highp>		highp_mat4x3;
typedef tmat4x4<float, lowp>		lowp_mat4;
typedef tmat4x4<float, mediump>		mediump_mat4;
typedef tmat4x4<float, highp>		highp_mat4;
typedef tmat4x4<float, lowp>		lowp_mat4x4;
typedef tmat4x4<float, mediump>		mediump_mat4x4;
typedef tmat4x4<float, highp>		highp_mat4x4;
#if(defined(GLM_PRECISION_LOWP_FLOAT))
typedef lowp_mat2x2			mat2x2;
typedef lowp_mat2x3			mat2x3;
typedef lowp_mat2x4			mat2x4;
typedef lowp_mat3x2			mat3x2;
typedef lowp_mat3x3			mat3x3;
typedef lowp_mat3x4			mat3x4;
typedef lowp_mat4x2			mat4x2;
typedef lowp_mat4x3			mat4x3;
typedef lowp_mat4x4			mat4x4;
#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
typedef mediump_mat2x2		mat2x2;
typedef mediump_mat2x3		mat2x3;
typedef mediump_mat2x4		mat2x4;
typedef mediump_mat3x2		mat3x2;
typedef mediump_mat3x3		mat3x3;
typedef mediump_mat3x4		mat3x4;
typedef mediump_mat4x2		mat4x2;
typedef mediump_mat4x3		mat4x3;
typedef mediump_mat4x4		mat4x4;
#else	
typedef highp_mat2x2			mat2x2;
typedef highp_mat2x3			mat2x3;
typedef highp_mat2x4			mat2x4;
typedef highp_mat3x2			mat3x2;
typedef highp_mat3x3			mat3x3;
typedef highp_mat3x4			mat3x4;
typedef highp_mat4x2			mat4x2;
typedef highp_mat4x3			mat4x3;
typedef highp_mat4x4			mat4x4;
#endif
typedef mat2x2					mat2;
typedef mat3x3					mat3;
typedef mat4x4					mat4;
typedef tmat2x2<double, lowp>		lowp_dmat2;
typedef tmat2x2<double, mediump>	mediump_dmat2;
typedef tmat2x2<double, highp>		highp_dmat2;
typedef tmat2x2<double, lowp>		lowp_dmat2x2;
typedef tmat2x2<double, mediump>	mediump_dmat2x2;
typedef tmat2x2<double, highp>		highp_dmat2x2;
typedef tmat2x3<double, lowp>		lowp_dmat2x3;
typedef tmat2x3<double, mediump>	mediump_dmat2x3;
typedef tmat2x3<double, highp>		highp_dmat2x3;
typedef tmat2x4<double, lowp>		lowp_dmat2x4;
typedef tmat2x4<double, mediump>	mediump_dmat2x4;
typedef tmat2x4<double, highp>		highp_dmat2x4;
typedef tmat3x2<double, lowp>		lowp_dmat3x2;
typedef tmat3x2<double, mediump>	mediump_dmat3x2;
typedef tmat3x2<double, highp>		highp_dmat3x2;
typedef tmat3x3<float, lowp>		lowp_dmat3;
typedef tmat3x3<double, mediump>	mediump_dmat3;
typedef tmat3x3<double, highp>		highp_dmat3;
typedef tmat3x3<double, lowp>		lowp_dmat3x3;
typedef tmat3x3<double, mediump>	mediump_dmat3x3;
typedef tmat3x3<double, highp>		highp_dmat3x3;
typedef tmat3x4<double, lowp>		lowp_dmat3x4;
typedef tmat3x4<double, mediump>	mediump_dmat3x4;
typedef tmat3x4<double, highp>		highp_dmat3x4;
typedef tmat4x2<double, lowp>		lowp_dmat4x2;
typedef tmat4x2<double, mediump>	mediump_dmat4x2;
typedef tmat4x2<double, highp>		highp_dmat4x2;
typedef tmat4x3<double, lowp>		lowp_dmat4x3;
typedef tmat4x3<double, mediump>	mediump_dmat4x3;
typedef tmat4x3<double, highp>		highp_dmat4x3;
typedef tmat4x4<double, lowp>		lowp_dmat4;
typedef tmat4x4<double, mediump>	mediump_dmat4;
typedef tmat4x4<double, highp>		highp_dmat4;
typedef tmat4x4<double, lowp>		lowp_dmat4x4;
typedef tmat4x4<double, mediump>	mediump_dmat4x4;
typedef tmat4x4<double, highp>		highp_dmat4x4;
#if(defined(GLM_PRECISION_LOWP_DOUBLE))
typedef lowp_dmat2x2		dmat2x2;
typedef lowp_dmat2x3		dmat2x3;
typedef lowp_dmat2x4		dmat2x4;
typedef lowp_dmat3x2		dmat3x2;
typedef lowp_dmat3x3		dmat3x3;
typedef lowp_dmat3x4		dmat3x4;
typedef lowp_dmat4x2		dmat4x2;
typedef lowp_dmat4x3		dmat4x3;
typedef lowp_dmat4x4		dmat4x4;
#elif(defined(GLM_PRECISION_MEDIUMP_DOUBLE))
typedef mediump_dmat2x2		dmat2x2;
typedef mediump_dmat2x3		dmat2x3;
typedef mediump_dmat2x4		dmat2x4;
typedef mediump_dmat3x2		dmat3x2;
typedef mediump_dmat3x3		dmat3x3;
typedef mediump_dmat3x4		dmat3x4;
typedef mediump_dmat4x2		dmat4x2;
typedef mediump_dmat4x3		dmat4x3;
typedef mediump_dmat4x4		dmat4x4;
#else 
typedef highp_dmat2x2		dmat2;
typedef highp_dmat3x3		dmat3;
typedef highp_dmat4x4		dmat4;
typedef highp_dmat2x2		dmat2x2;
typedef highp_dmat2x3		dmat2x3;
typedef highp_dmat2x4		dmat2x4;
typedef highp_dmat3x2		dmat3x2;
typedef highp_dmat3x3		dmat3x3;
typedef highp_dmat3x4		dmat3x4;
typedef highp_dmat4x2		dmat4x2;
typedef highp_dmat4x3		dmat4x3;
typedef highp_dmat4x4		dmat4x4;
#endif
}
