#pragma once
#include "precision.hpp"
#include "type_int.hpp"
namespace glm
{
template <typename T, precision P> struct tvec1;
template <typename T, precision P> struct tvec2;
template <typename T, precision P> struct tvec3;
template <typename T, precision P> struct tvec4;
typedef tvec1<float, highp>		highp_vec1_t;
typedef tvec1<float, mediump>	mediump_vec1_t;
typedef tvec1<float, lowp>		lowp_vec1_t;
typedef tvec1<double, highp>	highp_dvec1_t;
typedef tvec1<double, mediump>	mediump_dvec1_t;
typedef tvec1<double, lowp>		lowp_dvec1_t;
typedef tvec1<int, highp>		highp_ivec1_t;
typedef tvec1<int, mediump>		mediump_ivec1_t;
typedef tvec1<int, lowp>		lowp_ivec1_t;
typedef tvec1<uint, highp>		highp_uvec1_t;
typedef tvec1<uint, mediump>	mediump_uvec1_t;
typedef tvec1<uint, lowp>		lowp_uvec1_t;
typedef tvec1<bool, highp>		highp_bvec1_t;
typedef tvec1<bool, mediump>	mediump_bvec1_t;
typedef tvec1<bool, lowp>		lowp_bvec1_t;
typedef tvec2<float, highp>		highp_vec2;
typedef tvec2<float, mediump>	mediump_vec2;
typedef tvec2<float, lowp>		lowp_vec2;
typedef tvec2<double, highp>	highp_dvec2;
typedef tvec2<double, mediump>	mediump_dvec2;
typedef tvec2<double, lowp>		lowp_dvec2;
typedef tvec2<int, highp>		highp_ivec2;
typedef tvec2<int, mediump>		mediump_ivec2;
typedef tvec2<int, lowp>		lowp_ivec2;
typedef tvec2<uint, highp>		highp_uvec2;
typedef tvec2<uint, mediump>	mediump_uvec2;
typedef tvec2<uint, lowp>		lowp_uvec2;
typedef tvec2<bool, highp>		highp_bvec2;
typedef tvec2<bool, mediump>	mediump_bvec2;
typedef tvec2<bool, lowp>		lowp_bvec2;
typedef tvec3<float, highp>		highp_vec3;
typedef tvec3<float, mediump>	mediump_vec3;
typedef tvec3<float, lowp>		lowp_vec3;
typedef tvec3<double, highp>	highp_dvec3;
typedef tvec3<double, mediump>	mediump_dvec3;
typedef tvec3<double, lowp>		lowp_dvec3;
typedef tvec3<int, highp>		highp_ivec3;
typedef tvec3<int, mediump>		mediump_ivec3;
typedef tvec3<int, lowp>		lowp_ivec3;
typedef tvec3<uint, highp>		highp_uvec3;
typedef tvec3<uint, mediump>	mediump_uvec3;
typedef tvec3<uint, lowp>		lowp_uvec3;
typedef tvec3<bool, highp>		highp_bvec3;
typedef tvec3<bool, mediump>	mediump_bvec3;
typedef tvec3<bool, lowp>		lowp_bvec3;
typedef tvec4<float, highp>		highp_vec4;
typedef tvec4<float, mediump>	mediump_vec4;
typedef tvec4<float, lowp>		lowp_vec4;
typedef tvec4<double, highp>	highp_dvec4;
typedef tvec4<double, mediump>	mediump_dvec4;
typedef tvec4<double, lowp>		lowp_dvec4;
typedef tvec4<int, highp>		highp_ivec4;
typedef tvec4<int, mediump>		mediump_ivec4;
typedef tvec4<int, lowp>		lowp_ivec4;
typedef tvec4<uint, highp>		highp_uvec4;
typedef tvec4<uint, mediump>	mediump_uvec4;
typedef tvec4<uint, lowp>		lowp_uvec4;
typedef tvec4<bool, highp>		highp_bvec4;
typedef tvec4<bool, mediump>	mediump_bvec4;
typedef tvec4<bool, lowp>		lowp_bvec4;
#if(defined(GLM_PRECISION_LOWP_FLOAT))
typedef lowp_vec2			vec2;
typedef lowp_vec3			vec3;
typedef lowp_vec4			vec4;
#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
typedef mediump_vec2		vec2;
typedef mediump_vec3		vec3;
typedef mediump_vec4		vec4;
#else 
typedef highp_vec2			vec2;
typedef highp_vec3			vec3;
typedef highp_vec4			vec4;
#endif
#if(defined(GLM_PRECISION_LOWP_DOUBLE))
typedef lowp_dvec2			dvec2;
typedef lowp_dvec3			dvec3;
typedef lowp_dvec4			dvec4;
#elif(defined(GLM_PRECISION_MEDIUMP_DOUBLE))
typedef mediump_dvec2		dvec2;
typedef mediump_dvec3		dvec3;
typedef mediump_dvec4		dvec4;
#else 
typedef highp_dvec2			dvec2;
typedef highp_dvec3			dvec3;
typedef highp_dvec4			dvec4;
#endif
#if(defined(GLM_PRECISION_LOWP_INT))
typedef lowp_ivec2			ivec2;
typedef lowp_ivec3			ivec3;
typedef lowp_ivec4			ivec4;
#elif(defined(GLM_PRECISION_MEDIUMP_INT))
typedef mediump_ivec2		ivec2;
typedef mediump_ivec3		ivec3;
typedef mediump_ivec4		ivec4;
#else 
typedef highp_ivec2			ivec2;
typedef highp_ivec3			ivec3;
typedef highp_ivec4			ivec4;
#endif
#if(defined(GLM_PRECISION_LOWP_UINT))
typedef lowp_uvec2			uvec2;
typedef lowp_uvec3			uvec3;
typedef lowp_uvec4			uvec4;
#elif(defined(GLM_PRECISION_MEDIUMP_UINT))
typedef mediump_uvec2		uvec2;
typedef mediump_uvec3		uvec3;
typedef mediump_uvec4		uvec4;
#else 
typedef highp_uvec2			uvec2;
typedef highp_uvec3			uvec3;
typedef highp_uvec4			uvec4;
#endif
#if(defined(GLM_PRECISION_LOWP_BOOL))
typedef lowp_bvec2			bvec2;
typedef lowp_bvec3			bvec3;
typedef lowp_bvec4			bvec4;
#elif(defined(GLM_PRECISION_MEDIUMP_BOOL))
typedef mediump_bvec2		bvec2;
typedef mediump_bvec3		bvec3;
typedef mediump_bvec4		bvec4;
#else 
typedef highp_bvec2			bvec2;
typedef highp_bvec3			bvec3;
typedef highp_bvec4			bvec4;
#endif
}
