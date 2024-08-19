
#ifndef GLM_GTX_compatibility
#define GLM_GTX_compatibility GLM_VERSION

#include "../glm.hpp"  
#include "../gtc/half_float.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
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
template <typename T> GLM_FUNC_QUALIFIER detail::tvec2<T> lerp(const detail::tvec2<T>& x, const detail::tvec2<T>& y, T a){return mix(x, y, a);}							
template <typename T> GLM_FUNC_QUALIFIER detail::tvec3<T> lerp(const detail::tvec3<T>& x, const detail::tvec3<T>& y, T a){return mix(x, y, a);}							
template <typename T> GLM_FUNC_QUALIFIER detail::tvec4<T> lerp(const detail::tvec4<T>& x, const detail::tvec4<T>& y, T a){return mix(x, y, a);}							
template <typename T> GLM_FUNC_QUALIFIER detail::tvec2<T> lerp(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<T>& a){return mix(x, y, a);}	
template <typename T> GLM_FUNC_QUALIFIER detail::tvec3<T> lerp(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<T>& a){return mix(x, y, a);}	
template <typename T> GLM_FUNC_QUALIFIER detail::tvec4<T> lerp(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<T>& a){return mix(x, y, a);}	

template <typename T> GLM_FUNC_QUALIFIER T slerp(detail::tquat<T> const & x, detail::tquat<T> const & y, T const & a){return mix(x, y, a);} 

template <typename T> GLM_FUNC_QUALIFIER T saturate(T x){return clamp(x, T(0), T(1));}														
template <typename T> GLM_FUNC_QUALIFIER detail::tvec2<T> saturate(const detail::tvec2<T>& x){return clamp(x, T(0), T(1));}					
template <typename T> GLM_FUNC_QUALIFIER detail::tvec3<T> saturate(const detail::tvec3<T>& x){return clamp(x, T(0), T(1));}					
template <typename T> GLM_FUNC_QUALIFIER detail::tvec4<T> saturate(const detail::tvec4<T>& x){return clamp(x, T(0), T(1));}					

template <typename T> GLM_FUNC_QUALIFIER T atan2(T x, T y){return atan(x, y);}																
template <typename T> GLM_FUNC_QUALIFIER detail::tvec2<T> atan2(const detail::tvec2<T>& x, const detail::tvec2<T>& y){return atan(x, y);}	
template <typename T> GLM_FUNC_QUALIFIER detail::tvec3<T> atan2(const detail::tvec3<T>& x, const detail::tvec3<T>& y){return atan(x, y);}	
template <typename T> GLM_FUNC_QUALIFIER detail::tvec4<T> atan2(const detail::tvec4<T>& x, const detail::tvec4<T>& y){return atan(x, y);}	

template <typename genType> bool isfinite(genType const & x);											
template <typename valType> detail::tvec2<bool> isfinite(const detail::tvec2<valType>& x);				
template <typename valType> detail::tvec3<bool> isfinite(const detail::tvec3<valType>& x);				
template <typename valType> detail::tvec4<bool> isfinite(const detail::tvec4<valType>& x);				

typedef bool						bool1;			
typedef detail::tvec2<bool>			bool2;			
typedef detail::tvec3<bool>			bool3;			
typedef detail::tvec4<bool>			bool4;			

typedef bool						bool1x1;		
typedef detail::tmat2x2<bool>		bool2x2;		
typedef detail::tmat2x3<bool>		bool2x3;		
typedef detail::tmat2x4<bool>		bool2x4;		
typedef detail::tmat3x2<bool>		bool3x2;		
typedef detail::tmat3x3<bool>		bool3x3;		
typedef detail::tmat3x4<bool>		bool3x4;		
typedef detail::tmat4x2<bool>		bool4x2;		
typedef detail::tmat4x3<bool>		bool4x3;		
typedef detail::tmat4x4<bool>		bool4x4;		

typedef int							int1;			
typedef detail::tvec2<int>			int2;			
typedef detail::tvec3<int>			int3;			
typedef detail::tvec4<int>			int4;			

typedef int							int1x1;			
typedef detail::tmat2x2<int>		int2x2;			
typedef detail::tmat2x3<int>		int2x3;			
typedef detail::tmat2x4<int>		int2x4;			
typedef detail::tmat3x2<int>		int3x2;			
typedef detail::tmat3x3<int>		int3x3;			
typedef detail::tmat3x4<int>		int3x4;			
typedef detail::tmat4x2<int>		int4x2;			
typedef detail::tmat4x3<int>		int4x3;			
typedef detail::tmat4x4<int>		int4x4;			

typedef detail::half					half1;			
typedef detail::tvec2<detail::half>	half2;			
typedef detail::tvec3<detail::half>	half3;			
typedef detail::tvec4<detail::half>	half4;			

typedef detail::half					half1x1;		
typedef detail::tmat2x2<detail::half>	half2x2;		
typedef detail::tmat2x3<detail::half>	half2x3;		
typedef detail::tmat2x4<detail::half>	half2x4;		
typedef detail::tmat3x2<detail::half>	half3x2;		
typedef detail::tmat3x3<detail::half>	half3x3;		
typedef detail::tmat3x4<detail::half>	half3x4;		
typedef detail::tmat4x2<detail::half>	half4x2;		
typedef detail::tmat4x3<detail::half>	half4x3;		
typedef detail::tmat4x4<detail::half>	half4x4;		

typedef float						float1;			
typedef detail::tvec2<float>		float2;			
typedef detail::tvec3<float>		float3;			
typedef detail::tvec4<float>		float4;			

typedef float						float1x1;		
typedef detail::tmat2x2<float>		float2x2;		
typedef detail::tmat2x3<float>		float2x3;		
typedef detail::tmat2x4<float>		float2x4;		
typedef detail::tmat3x2<float>		float3x2;		
typedef detail::tmat3x3<float>		float3x3;		
typedef detail::tmat3x4<float>		float3x4;		
typedef detail::tmat4x2<float>		float4x2;		
typedef detail::tmat4x3<float>		float4x3;		
typedef detail::tmat4x4<float>		float4x4;		

typedef double						double1;		
typedef detail::tvec2<double>		double2;		
typedef detail::tvec3<double>		double3;		
typedef detail::tvec4<double>		double4;		

typedef double						double1x1;		
typedef detail::tmat2x2<double>		double2x2;		
typedef detail::tmat2x3<double>		double2x3;		
typedef detail::tmat2x4<double>		double2x4;		
typedef detail::tmat3x2<double>		double3x2;		
typedef detail::tmat3x3<double>		double3x3;		
typedef detail::tmat3x4<double>		double3x4;		
typedef detail::tmat4x2<double>		double4x2;		
typedef detail::tmat4x3<double>		double4x3;		
typedef detail::tmat4x4<double>		double4x4;		

}

#include "compatibility.inl"

#endif

