
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_handed_coordinate_space is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_handed_coordinate_space extension included")
#	endif
#endif

namespace glm
{

template<typename T, qualifier Q>
GLM_FUNC_DECL bool rightHanded(
vec<3, T, Q> const& tangent,
vec<3, T, Q> const& binormal,
vec<3, T, Q> const& normal);

template<typename T, qualifier Q>
GLM_FUNC_DECL bool leftHanded(
vec<3, T, Q> const& tangent,
vec<3, T, Q> const& binormal,
vec<3, T, Q> const& normal);

}

#include "handed_coordinate_space.inl"
