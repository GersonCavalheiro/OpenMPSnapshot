#pragma once
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_matrix_integer extension included")
#endif
namespace glm
{
typedef tmat2x2<int, highp>				highp_imat2;
typedef tmat3x3<int, highp>				highp_imat3;
typedef tmat4x4<int, highp>				highp_imat4;
typedef tmat2x2<int, highp>				highp_imat2x2;
typedef tmat2x3<int, highp>				highp_imat2x3;
typedef tmat2x4<int, highp>				highp_imat2x4;
typedef tmat3x2<int, highp>				highp_imat3x2;
typedef tmat3x3<int, highp>				highp_imat3x3;
typedef tmat3x4<int, highp>				highp_imat3x4;
typedef tmat4x2<int, highp>				highp_imat4x2;
typedef tmat4x3<int, highp>				highp_imat4x3;
typedef tmat4x4<int, highp>				highp_imat4x4;
typedef tmat2x2<int, mediump>			mediump_imat2;
typedef tmat3x3<int, mediump>			mediump_imat3;
typedef tmat4x4<int, mediump>			mediump_imat4;
typedef tmat2x2<int, mediump>			mediump_imat2x2;
typedef tmat2x3<int, mediump>			mediump_imat2x3;
typedef tmat2x4<int, mediump>			mediump_imat2x4;
typedef tmat3x2<int, mediump>			mediump_imat3x2;
typedef tmat3x3<int, mediump>			mediump_imat3x3;
typedef tmat3x4<int, mediump>			mediump_imat3x4;
typedef tmat4x2<int, mediump>			mediump_imat4x2;
typedef tmat4x3<int, mediump>			mediump_imat4x3;
typedef tmat4x4<int, mediump>			mediump_imat4x4;
typedef tmat2x2<int, lowp>				lowp_imat2;
typedef tmat3x3<int, lowp>				lowp_imat3;
typedef tmat4x4<int, lowp>				lowp_imat4;
typedef tmat2x2<int, lowp>				lowp_imat2x2;
typedef tmat2x3<int, lowp>				lowp_imat2x3;
typedef tmat2x4<int, lowp>				lowp_imat2x4;
typedef tmat3x2<int, lowp>				lowp_imat3x2;
typedef tmat3x3<int, lowp>				lowp_imat3x3;
typedef tmat3x4<int, lowp>				lowp_imat3x4;
typedef tmat4x2<int, lowp>				lowp_imat4x2;
typedef tmat4x3<int, lowp>				lowp_imat4x3;
typedef tmat4x4<int, lowp>				lowp_imat4x4;
typedef tmat2x2<uint, highp>				highp_umat2;	
typedef tmat3x3<uint, highp>				highp_umat3;
typedef tmat4x4<uint, highp>				highp_umat4;
typedef tmat2x2<uint, highp>				highp_umat2x2;
typedef tmat2x3<uint, highp>				highp_umat2x3;
typedef tmat2x4<uint, highp>				highp_umat2x4;
typedef tmat3x2<uint, highp>				highp_umat3x2;
typedef tmat3x3<uint, highp>				highp_umat3x3;
typedef tmat3x4<uint, highp>				highp_umat3x4;
typedef tmat4x2<uint, highp>				highp_umat4x2;
typedef tmat4x3<uint, highp>				highp_umat4x3;
typedef tmat4x4<uint, highp>				highp_umat4x4;
typedef tmat2x2<uint, mediump>			mediump_umat2;
typedef tmat3x3<uint, mediump>			mediump_umat3;
typedef tmat4x4<uint, mediump>			mediump_umat4;
typedef tmat2x2<uint, mediump>			mediump_umat2x2;
typedef tmat2x3<uint, mediump>			mediump_umat2x3;
typedef tmat2x4<uint, mediump>			mediump_umat2x4;
typedef tmat3x2<uint, mediump>			mediump_umat3x2;
typedef tmat3x3<uint, mediump>			mediump_umat3x3;
typedef tmat3x4<uint, mediump>			mediump_umat3x4;
typedef tmat4x2<uint, mediump>			mediump_umat4x2;
typedef tmat4x3<uint, mediump>			mediump_umat4x3;
typedef tmat4x4<uint, mediump>			mediump_umat4x4;
typedef tmat2x2<uint, lowp>				lowp_umat2;
typedef tmat3x3<uint, lowp>				lowp_umat3;
typedef tmat4x4<uint, lowp>				lowp_umat4;
typedef tmat2x2<uint, lowp>				lowp_umat2x2;
typedef tmat2x3<uint, lowp>				lowp_umat2x3;
typedef tmat2x4<uint, lowp>				lowp_umat2x4;
typedef tmat3x2<uint, lowp>				lowp_umat3x2;
typedef tmat3x3<uint, lowp>				lowp_umat3x3;
typedef tmat3x4<uint, lowp>				lowp_umat3x4;
typedef tmat4x2<uint, lowp>				lowp_umat4x2;
typedef tmat4x3<uint, lowp>				lowp_umat4x3;
typedef tmat4x4<uint, lowp>				lowp_umat4x4;
#if(defined(GLM_PRECISION_HIGHP_INT))
typedef highp_imat2								imat2;
typedef highp_imat3								imat3;
typedef highp_imat4								imat4;
typedef highp_imat2x2							imat2x2;
typedef highp_imat2x3							imat2x3;
typedef highp_imat2x4							imat2x4;
typedef highp_imat3x2							imat3x2;
typedef highp_imat3x3							imat3x3;
typedef highp_imat3x4							imat3x4;
typedef highp_imat4x2							imat4x2;
typedef highp_imat4x3							imat4x3;
typedef highp_imat4x4							imat4x4;
#elif(defined(GLM_PRECISION_LOWP_INT))
typedef lowp_imat2								imat2;
typedef lowp_imat3								imat3;
typedef lowp_imat4								imat4;
typedef lowp_imat2x2							imat2x2;
typedef lowp_imat2x3							imat2x3;
typedef lowp_imat2x4							imat2x4;
typedef lowp_imat3x2							imat3x2;
typedef lowp_imat3x3							imat3x3;
typedef lowp_imat3x4							imat3x4;
typedef lowp_imat4x2							imat4x2;
typedef lowp_imat4x3							imat4x3;
typedef lowp_imat4x4							imat4x4;
#else 
typedef mediump_imat2							imat2;
typedef mediump_imat3							imat3;
typedef mediump_imat4							imat4;
typedef mediump_imat2x2							imat2x2;
typedef mediump_imat2x3							imat2x3;
typedef mediump_imat2x4							imat2x4;
typedef mediump_imat3x2							imat3x2;
typedef mediump_imat3x3							imat3x3;
typedef mediump_imat3x4							imat3x4;
typedef mediump_imat4x2							imat4x2;
typedef mediump_imat4x3							imat4x3;
typedef mediump_imat4x4							imat4x4;
#endif
#if(defined(GLM_PRECISION_HIGHP_UINT))
typedef highp_umat2								umat2;
typedef highp_umat3								umat3;
typedef highp_umat4								umat4;
typedef highp_umat2x2							umat2x2;
typedef highp_umat2x3							umat2x3;
typedef highp_umat2x4							umat2x4;
typedef highp_umat3x2							umat3x2;
typedef highp_umat3x3							umat3x3;
typedef highp_umat3x4							umat3x4;
typedef highp_umat4x2							umat4x2;
typedef highp_umat4x3							umat4x3;
typedef highp_umat4x4							umat4x4;
#elif(defined(GLM_PRECISION_LOWP_UINT))
typedef lowp_umat2								umat2;
typedef lowp_umat3								umat3;
typedef lowp_umat4								umat4;
typedef lowp_umat2x2							umat2x2;
typedef lowp_umat2x3							umat2x3;
typedef lowp_umat2x4							umat2x4;
typedef lowp_umat3x2							umat3x2;
typedef lowp_umat3x3							umat3x3;
typedef lowp_umat3x4							umat3x4;
typedef lowp_umat4x2							umat4x2;
typedef lowp_umat4x3							umat4x3;
typedef lowp_umat4x4							umat4x4;
#else 
typedef mediump_umat2							umat2;
typedef mediump_umat3							umat3;
typedef mediump_umat4							umat4;
typedef mediump_umat2x2							umat2x2;
typedef mediump_umat2x3							umat2x3;
typedef mediump_umat2x4							umat2x4;
typedef mediump_umat3x2							umat3x2;
typedef mediump_umat3x3							umat3x3;
typedef mediump_umat3x4							umat3x4;
typedef mediump_umat4x2							umat4x2;
typedef mediump_umat4x3							umat4x3;
typedef mediump_umat4x4							umat4x4;
#endif
}
