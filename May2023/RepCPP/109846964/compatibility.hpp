
#ifndef GLM_GTX_compatibility
#define GLM_GTX_compatibility

#include "../glm.hpp"  
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_compatibility extension included")
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
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec2<T, P> lerp(const detail::tvec2<T, P>& x, const detail::tvec2<T, P>& y, T a){return mix(x, y, a);}							

template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec3<T, P> lerp(const detail::tvec3<T, P>& x, const detail::tvec3<T, P>& y, T a){return mix(x, y, a);}							
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec4<T, P> lerp(const detail::tvec4<T, P>& x, const detail::tvec4<T, P>& y, T a){return mix(x, y, a);}							
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec2<T, P> lerp(const detail::tvec2<T, P>& x, const detail::tvec2<T, P>& y, const detail::tvec2<T, P>& a){return mix(x, y, a);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec3<T, P> lerp(const detail::tvec3<T, P>& x, const detail::tvec3<T, P>& y, const detail::tvec3<T, P>& a){return mix(x, y, a);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec4<T, P> lerp(const detail::tvec4<T, P>& x, const detail::tvec4<T, P>& y, const detail::tvec4<T, P>& a){return mix(x, y, a);}	

template <typename T, precision P> GLM_FUNC_QUALIFIER T slerp(detail::tquat<T, P> const & x, detail::tquat<T, P> const & y, T const & a){return mix(x, y, a);} 

template <typename T, precision P> GLM_FUNC_QUALIFIER T saturate(T x){return clamp(x, T(0), T(1));}														
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec2<T, P> saturate(const detail::tvec2<T, P>& x){return clamp(x, T(0), T(1));}					
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec3<T, P> saturate(const detail::tvec3<T, P>& x){return clamp(x, T(0), T(1));}					
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec4<T, P> saturate(const detail::tvec4<T, P>& x){return clamp(x, T(0), T(1));}					

template <typename T, precision P> GLM_FUNC_QUALIFIER T atan2(T x, T y){return atan(x, y);}																
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec2<T, P> atan2(const detail::tvec2<T, P>& x, const detail::tvec2<T, P>& y){return atan(x, y);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec3<T, P> atan2(const detail::tvec3<T, P>& x, const detail::tvec3<T, P>& y){return atan(x, y);}	
template <typename T, precision P> GLM_FUNC_QUALIFIER detail::tvec4<T, P> atan2(const detail::tvec4<T, P>& x, const detail::tvec4<T, P>& y){return atan(x, y);}	

template <typename genType> GLM_FUNC_DECL bool isfinite(genType const & x);											
template <typename T, precision P> GLM_FUNC_DECL detail::tvec2<bool, P> isfinite(const detail::tvec2<T, P>& x);				
template <typename T, precision P> GLM_FUNC_DECL detail::tvec3<bool, P> isfinite(const detail::tvec3<T, P>& x);				
template <typename T, precision P> GLM_FUNC_DECL detail::tvec4<bool, P> isfinite(const detail::tvec4<T, P>& x);				

typedef bool						bool1;			
typedef detail::tvec2<bool, highp>			bool2;			
typedef detail::tvec3<bool, highp>			bool3;			
typedef detail::tvec4<bool, highp>			bool4;			

typedef bool						bool1x1;		
typedef detail::tmat2x2<bool, highp>		bool2x2;		
typedef detail::tmat2x3<bool, highp>		bool2x3;		
typedef detail::tmat2x4<bool, highp>		bool2x4;		
typedef detail::tmat3x2<bool, highp>		bool3x2;		
typedef detail::tmat3x3<bool, highp>		bool3x3;		
typedef detail::tmat3x4<bool, highp>		bool3x4;		
typedef detail::tmat4x2<bool, highp>		bool4x2;		
typedef detail::tmat4x3<bool, highp>		bool4x3;		
typedef detail::tmat4x4<bool, highp>		bool4x4;		

typedef int							int1;			
typedef detail::tvec2<int, highp>			int2;			
typedef detail::tvec3<int, highp>			int3;			
typedef detail::tvec4<int, highp>			int4;			

typedef int							int1x1;			
typedef detail::tmat2x2<int, highp>		int2x2;			
typedef detail::tmat2x3<int, highp>		int2x3;			
typedef detail::tmat2x4<int, highp>		int2x4;			
typedef detail::tmat3x2<int, highp>		int3x2;			
typedef detail::tmat3x3<int, highp>		int3x3;			
typedef detail::tmat3x4<int, highp>		int3x4;			
typedef detail::tmat4x2<int, highp>		int4x2;			
typedef detail::tmat4x3<int, highp>		int4x3;			
typedef detail::tmat4x4<int, highp>		int4x4;			

typedef float						float1;			
typedef detail::tvec2<float, highp>		float2;			
typedef detail::tvec3<float, highp>		float3;			
typedef detail::tvec4<float, highp>		float4;			

typedef float						float1x1;		
typedef detail::tmat2x2<float, highp>		float2x2;		
typedef detail::tmat2x3<float, highp>		float2x3;		
typedef detail::tmat2x4<float, highp>		float2x4;		
typedef detail::tmat3x2<float, highp>		float3x2;		
typedef detail::tmat3x3<float, highp>		float3x3;		
typedef detail::tmat3x4<float, highp>		float3x4;		
typedef detail::tmat4x2<float, highp>		float4x2;		
typedef detail::tmat4x3<float, highp>		float4x3;		
typedef detail::tmat4x4<float, highp>		float4x4;		

typedef double						double1;		
typedef detail::tvec2<double, highp>		double2;		
typedef detail::tvec3<double, highp>		double3;		
typedef detail::tvec4<double, highp>		double4;		

typedef double						double1x1;		
typedef detail::tmat2x2<double, highp>		double2x2;		
typedef detail::tmat2x3<double, highp>		double2x3;		
typedef detail::tmat2x4<double, highp>		double2x4;		
typedef detail::tmat3x2<double, highp>		double3x2;		
typedef detail::tmat3x3<double, highp>		double3x3;		
typedef detail::tmat3x4<double, highp>		double3x4;		
typedef detail::tmat4x2<double, highp>		double4x2;		
typedef detail::tmat4x3<double, highp>		double4x3;		
typedef detail::tmat4x4<double, highp>		double4x4;		

}

#include "compatibility.inl"

#endif

