
#pragma once

#include "../gtx/fast_square_root.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_normalize_dot is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_normalize_dot extension included")
#	endif
#endif

namespace glm
{

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL T normalizeDot(vec<L, T, Q> const& x, vec<L, T, Q> const& y);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL T fastNormalizeDot(vec<L, T, Q> const& x, vec<L, T, Q> const& y);

}

#include "normalize_dot.inl"
