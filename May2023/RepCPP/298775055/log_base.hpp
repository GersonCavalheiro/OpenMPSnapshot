
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_log_base is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_log_base extension included")
#	endif
#endif

namespace glm
{

template<typename genType>
GLM_FUNC_DECL genType log(
genType const& x,
genType const& base);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> sign(
vec<L, T, Q> const& x,
vec<L, T, Q> const& base);

}

#include "log_base.inl"
