
#ifndef GLM_GTX_color_cast
#define GLM_GTX_color_cast GLM_VERSION

#include "../glm.hpp"
#include "../gtx/number_precision.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_color_cast extension included")
#endif

namespace glm
{

template <typename valType> uint8 u8channel_cast(valType a);

template <typename valType>	uint16 u16channel_cast(valType a);

template <typename T> uint32 u32_rgbx_cast(const detail::tvec3<T>& c);		
template <typename T> uint32 u32_xrgb_cast(const detail::tvec3<T>& c);		
template <typename T> uint32 u32_bgrx_cast(const detail::tvec3<T>& c);		
template <typename T> uint32 u32_xbgr_cast(const detail::tvec3<T>& c);		

template <typename T> uint32 u32_rgba_cast(const detail::tvec4<T>& c);		
template <typename T> uint32 u32_argb_cast(const detail::tvec4<T>& c);		
template <typename T> uint32 u32_bgra_cast(const detail::tvec4<T>& c);		
template <typename T> uint32 u32_abgr_cast(const detail::tvec4<T>& c);		

template <typename T> uint64 u64_rgbx_cast(const detail::tvec3<T>& c);		
template <typename T> uint64 u64_xrgb_cast(const detail::tvec3<T>& c);		
template <typename T> uint64 u64_bgrx_cast(const detail::tvec3<T>& c);		
template <typename T> uint64 u64_xbgr_cast(const detail::tvec3<T>& c);		

template <typename T> uint64 u64_rgba_cast(const detail::tvec4<T>& c);		
template <typename T> uint64 u64_argb_cast(const detail::tvec4<T>& c);		
template <typename T> uint64 u64_bgra_cast(const detail::tvec4<T>& c);		
template <typename T> uint64 u64_abgr_cast(const detail::tvec4<T>& c);		

template <typename T> f16 f16_channel_cast(T a);	

template <typename T> f16vec3 f16_rgbx_cast(T c);		
template <typename T> f16vec3 f16_xrgb_cast(T c);		
template <typename T> f16vec3 f16_bgrx_cast(T c);		
template <typename T> f16vec3 f16_xbgr_cast(T c);		

template <typename T> f16vec4 f16_rgba_cast(T c);		
template <typename T> f16vec4 f16_argb_cast(T c);		
template <typename T> f16vec4 f16_bgra_cast(T c);		
template <typename T> f16vec4 f16_abgr_cast(T c);		

template <typename T> f32 f32_channel_cast(T a);		

template <typename T> f32vec3 f32_rgbx_cast(T c);		
template <typename T> f32vec3 f32_xrgb_cast(T c);		
template <typename T> f32vec3 f32_bgrx_cast(T c);		
template <typename T> f32vec3 f32_xbgr_cast(T c);		

template <typename T> f32vec4 f32_rgba_cast(T c);		
template <typename T> f32vec4 f32_argb_cast(T c);		
template <typename T> f32vec4 f32_bgra_cast(T c);		
template <typename T> f32vec4 f32_abgr_cast(T c);		

template <typename T> f64 f64_channel_cast(T a);		

template <typename T> f64vec3 f64_rgbx_cast(T c);		
template <typename T> f64vec3 f64_xrgb_cast(T c);		
template <typename T> f64vec3 f64_bgrx_cast(T c);		
template <typename T> f64vec3 f64_xbgr_cast(T c);		

template <typename T> f64vec4 f64_rgba_cast(T c);		
template <typename T> f64vec4 f64_argb_cast(T c);		
template <typename T> f64vec4 f64_bgra_cast(T c);		
template <typename T> f64vec4 f64_abgr_cast(T c);		

}

#include "color_cast.inl"

#endif
