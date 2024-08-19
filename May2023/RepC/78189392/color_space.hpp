#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_color_space extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rgbColor(
tvec3<T, P> const & hsvValue);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> hsvColor(
tvec3<T, P> const & rgbValue);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> saturation(
T const s);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> saturation(
T const s,
tvec3<T, P> const & color);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> saturation(
T const s,
tvec4<T, P> const & color);
template <typename T, precision P>
GLM_FUNC_DECL T luminosity(
tvec3<T, P> const & color);
}
#include "color_space.inl"
