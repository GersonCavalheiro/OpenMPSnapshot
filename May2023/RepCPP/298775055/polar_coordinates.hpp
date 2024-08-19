
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_polar_coordinates is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_polar_coordinates extension included")
#	endif
#endif

namespace glm
{

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> polar(
vec<3, T, Q> const& euclidean);

template<typename T, qualifier Q>
GLM_FUNC_DECL vec<3, T, Q> euclidean(
vec<2, T, Q> const& polar);

}

#include "polar_coordinates.inl"
