
#ifndef GLM_GTX_matrix_operation
#define GLM_GTX_matrix_operation

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_operation extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x2<T, P> diagonal2x2(
detail::tvec2<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x3<T, P> diagonal2x3(
detail::tvec2<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x4<T, P> diagonal2x4(
detail::tvec2<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x2<T, P> diagonal3x2(
detail::tvec2<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> diagonal3x3(
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x4<T, P> diagonal3x4(
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x2<T, P> diagonal4x2(
detail::tvec2<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x3<T, P> diagonal4x3(
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> diagonal4x4(
detail::tvec4<T, P> const & v);

}

#include "matrix_operation.inl"

#endif
