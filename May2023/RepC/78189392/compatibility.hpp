#pragma once
#include "../glm.hpp"
#include "../gtc/quaternion.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_compatibility extension included")
#endif
#if(GLM_COMPILER & GLM_COMPILER_VC)
#	include <cfloat>
#elif(GLM_COMPILER & GLM_COMPILER_GCC)
#	include <cmath>
#	if(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
#		undef isfinite
#	endif
#endif
namespace glm
{
template <typename T> GLM_FUNC_QUALIFIER T lerp(T x, T y, T a){return mix(x, y, a);}																					
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> lerp(const tvec2<T, P>& x, const tvec2<T, P>& y, T a){return mix(x, y, a);}							
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> lerp(const tvec3<T, P>& x, const tvec3<T, P>& y, T a){return mix(x, y, a);}							
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> lerp(const tvec4<T, P>& x, const tvec4<T, P>& y, T a){return mix(x, y, a);}							
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> lerp(const tvec2<T, P>& x, const tvec2<T, P>& y, const tvec2<T, P>& a){return mix(x, y, a);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> lerp(const tvec3<T, P>& x, const tvec3<T, P>& y, const tvec3<T, P>& a){return mix(x, y, a);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> lerp(const tvec4<T, P>& x, const tvec4<T, P>& y, const tvec4<T, P>& a){return mix(x, y, a);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER T saturate(T x){return clamp(x, T(0), T(1));}														
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> saturate(const tvec2<T, P>& x){return clamp(x, T(0), T(1));}					
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> saturate(const tvec3<T, P>& x){return clamp(x, T(0), T(1));}					
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> saturate(const tvec4<T, P>& x){return clamp(x, T(0), T(1));}					
template <typename T, precision P> GLM_FUNC_QUALIFIER T atan2(T x, T y){return atan(x, y);}																
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec2<T, P> atan2(const tvec2<T, P>& x, const tvec2<T, P>& y){return atan(x, y);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec3<T, P> atan2(const tvec3<T, P>& x, const tvec3<T, P>& y){return atan(x, y);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER tvec4<T, P> atan2(const tvec4<T, P>& x, const tvec4<T, P>& y){return atan(x, y);}	
template <typename genType> GLM_FUNC_DECL bool isfinite(genType const & x);											
template <typename T, precision P> GLM_FUNC_DECL tvec1<bool, P> isfinite(const tvec1<T, P>& x);				
template <typename T, precision P> GLM_FUNC_DECL tvec2<bool, P> isfinite(const tvec2<T, P>& x);				
template <typename T, precision P> GLM_FUNC_DECL tvec3<bool, P> isfinite(const tvec3<T, P>& x);				
template <typename T, precision P> GLM_FUNC_DECL tvec4<bool, P> isfinite(const tvec4<T, P>& x);				
typedef bool						bool1;			
typedef tvec2<bool, highp>			bool2;			
typedef tvec3<bool, highp>			bool3;			
typedef tvec4<bool, highp>			bool4;			
typedef bool						bool1x1;		
typedef tmat2x2<bool, highp>		bool2x2;		
typedef tmat2x3<bool, highp>		bool2x3;		
typedef tmat2x4<bool, highp>		bool2x4;		
typedef tmat3x2<bool, highp>		bool3x2;		
typedef tmat3x3<bool, highp>		bool3x3;		
typedef tmat3x4<bool, highp>		bool3x4;		
typedef tmat4x2<bool, highp>		bool4x2;		
typedef tmat4x3<bool, highp>		bool4x3;		
typedef tmat4x4<bool, highp>		bool4x4;		
typedef int							int1;			
typedef tvec2<int, highp>			int2;			
typedef tvec3<int, highp>			int3;			
typedef tvec4<int, highp>			int4;			
typedef int							int1x1;			
typedef tmat2x2<int, highp>		int2x2;			
typedef tmat2x3<int, highp>		int2x3;			
typedef tmat2x4<int, highp>		int2x4;			
typedef tmat3x2<int, highp>		int3x2;			
typedef tmat3x3<int, highp>		int3x3;			
typedef tmat3x4<int, highp>		int3x4;			
typedef tmat4x2<int, highp>		int4x2;			
typedef tmat4x3<int, highp>		int4x3;			
typedef tmat4x4<int, highp>		int4x4;			
typedef float						float1;			
typedef tvec2<float, highp>		float2;			
typedef tvec3<float, highp>		float3;			
typedef tvec4<float, highp>		float4;			
typedef float						float1x1;		
typedef tmat2x2<float, highp>		float2x2;		
typedef tmat2x3<float, highp>		float2x3;		
typedef tmat2x4<float, highp>		float2x4;		
typedef tmat3x2<float, highp>		float3x2;		
typedef tmat3x3<float, highp>		float3x3;		
typedef tmat3x4<float, highp>		float3x4;		
typedef tmat4x2<float, highp>		float4x2;		
typedef tmat4x3<float, highp>		float4x3;		
typedef tmat4x4<float, highp>		float4x4;		
typedef double						double1;		
typedef tvec2<double, highp>		double2;		
typedef tvec3<double, highp>		double3;		
typedef tvec4<double, highp>		double4;		
typedef double						double1x1;		
typedef tmat2x2<double, highp>		double2x2;		
typedef tmat2x3<double, highp>		double2x3;		
typedef tmat2x4<double, highp>		double2x4;		
typedef tmat3x2<double, highp>		double3x2;		
typedef tmat3x3<double, highp>		double3x3;		
typedef tmat3x4<double, highp>		double3x4;		
typedef tmat4x2<double, highp>		double4x2;		
typedef tmat4x3<double, highp>		double4x3;		
typedef tmat4x4<double, highp>		double4x4;		
}
#include "compatibility.inl"
