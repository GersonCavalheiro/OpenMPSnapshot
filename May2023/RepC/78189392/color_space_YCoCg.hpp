#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_color_space_YCoCg extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rgb2YCoCg(
tvec3<T, P> const & rgbColor);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> YCoCg2rgb(
tvec3<T, P> const & YCoCgColor);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rgb2YCoCgR(
tvec3<T, P> const & rgbColor);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> YCoCgR2rgb(
tvec3<T, P> const & YCoCgColor);
}
#include "color_space_YCoCg.inl"
