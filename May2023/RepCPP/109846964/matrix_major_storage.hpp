
#ifndef GLM_GTX_matrix_major_storage
#define GLM_GTX_matrix_major_storage

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_major_storage extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x2<T, P> rowMajor2(
detail::tvec2<T, P> const & v1, 
detail::tvec2<T, P> const & v2);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x2<T, P> rowMajor2(
detail::tmat2x2<T, P> const & m);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> rowMajor3(
detail::tvec3<T, P> const & v1, 
detail::tvec3<T, P> const & v2, 
detail::tvec3<T, P> const & v3);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> rowMajor3(
detail::tmat3x3<T, P> const & m);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> rowMajor4(
detail::tvec4<T, P> const & v1, 
detail::tvec4<T, P> const & v2,
detail::tvec4<T, P> const & v3, 
detail::tvec4<T, P> const & v4);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> rowMajor4(
detail::tmat4x4<T, P> const & m);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x2<T, P> colMajor2(
detail::tvec2<T, P> const & v1, 
detail::tvec2<T, P> const & v2);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x2<T, P> colMajor2(
detail::tmat2x2<T, P> const & m);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> colMajor3(
detail::tvec3<T, P> const & v1, 
detail::tvec3<T, P> const & v2, 
detail::tvec3<T, P> const & v3);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> colMajor3(
detail::tmat3x3<T, P> const & m);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> colMajor4(
detail::tvec4<T, P> const & v1, 
detail::tvec4<T, P> const & v2, 
detail::tvec4<T, P> const & v3, 
detail::tvec4<T, P> const & v4);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat4x4<T, P> colMajor4(
detail::tmat4x4<T, P> const & m);

}

#include "matrix_major_storage.inl"

#endif
