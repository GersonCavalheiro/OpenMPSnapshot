#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_matrix_cross_product extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> matrixCross3(
tvec3<T, P> const & x);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> matrixCross4(
tvec3<T, P> const & x);
}
#include "matrix_cross_product.inl"
