
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_color_space_YCoCg is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_color_space_YCoCg extension included")
#	endif
#endif

namespace glm
{

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> rgb2YCoCg(
vec<3, T, Q> const& rgbColor);

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> YCoCg2rgb(
vec<3, T, Q> const& YCoCgColor);

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> rgb2YCoCgR(
vec<3, T, Q> const& rgbColor);

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> YCoCgR2rgb(
vec<3, T, Q> const& YCoCgColor);

}

#include "color_space_YCoCg.inl"
