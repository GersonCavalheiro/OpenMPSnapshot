#pragma once
#include "../mat3x3.hpp"
#include "../vec2.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_matrix_transform_2d extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_QUALIFIER tmat3x3<T, P> translate(
tmat3x3<T, P> const & m,
tvec2<T, P> const & v);
template <typename T, precision P> 
GLM_FUNC_QUALIFIER tmat3x3<T, P> rotate(
tmat3x3<T, P> const & m,
T angle);
template <typename T, precision P> 
GLM_FUNC_QUALIFIER tmat3x3<T, P> scale(
tmat3x3<T, P> const & m,
tvec2<T, P> const & v);
template <typename T, precision P> 
GLM_FUNC_QUALIFIER tmat3x3<T, P> shearX(
tmat3x3<T, P> const & m,
T y);
template <typename T, precision P> 
GLM_FUNC_QUALIFIER tmat3x3<T, P> shearY(
tmat3x3<T, P> const & m,
T x);
}
#include "matrix_transform_2d.inl"
