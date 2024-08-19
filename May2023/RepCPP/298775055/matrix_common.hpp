
#pragma once

#include "../detail/qualifier.hpp"
#include "../detail/_fixes.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_transform extension included")
#endif

namespace glm
{

template<length_t C, length_t R, typename T, typename U, qualifier Q>
GLM_FUNC_DECL mat<C, R, T, Q> mix(mat<C, R, T, Q> const& x, mat<C, R, T, Q> const& y, mat<C, R, U, Q> const& a);

template<length_t C, length_t R, typename T, typename U, qualifier Q>
GLM_FUNC_DECL mat<C, R, T, Q> mix(mat<C, R, T, Q> const& x, mat<C, R, T, Q> const& y, U a);

}

#include "matrix_common.inl"
