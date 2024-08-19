
#ifndef GLM_GTX_matrix_transform_2d
#define GLM_GTX_matrix_transform_2d

#include "../mat3x3.hpp"
#include "../vec2.hpp"


#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_transform_2d extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_QUALIFIER detail::tmat3x3<T, P> translate(
detail::tmat3x3<T, P> const & m,
detail::tvec2<T, P> const & v);

template <typename T, precision P> 
GLM_FUNC_QUALIFIER detail::tmat3x3<T, P> rotate(
detail::tmat3x3<T, P> const & m,
T const & angle);

template <typename T, precision P> 
GLM_FUNC_QUALIFIER detail::tmat3x3<T, P> scale(
detail::tmat3x3<T, P> const & m,
detail::tvec2<T, P> const & v);

template <typename T, precision P> 
GLM_FUNC_QUALIFIER detail::tmat3x3<T, P> shearX(
detail::tmat3x3<T, P> const & m,
T const & y);

template <typename T, precision P> 
GLM_FUNC_QUALIFIER detail::tmat3x3<T, P> shearY(
detail::tmat3x3<T, P> const & m,
T const & x);

}

#include "matrix_transform_2d.inl"

#endif
