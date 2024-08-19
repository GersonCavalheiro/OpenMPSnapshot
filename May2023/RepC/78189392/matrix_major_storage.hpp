#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_matrix_major_storage extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tmat2x2<T, P> rowMajor2(
tvec2<T, P> const & v1, 
tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x2<T, P> rowMajor2(
tmat2x2<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> rowMajor3(
tvec3<T, P> const & v1, 
tvec3<T, P> const & v2, 
tvec3<T, P> const & v3);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> rowMajor3(
tmat3x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> rowMajor4(
tvec4<T, P> const & v1, 
tvec4<T, P> const & v2,
tvec4<T, P> const & v3, 
tvec4<T, P> const & v4);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> rowMajor4(
tmat4x4<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x2<T, P> colMajor2(
tvec2<T, P> const & v1, 
tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x2<T, P> colMajor2(
tmat2x2<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> colMajor3(
tvec3<T, P> const & v1, 
tvec3<T, P> const & v2, 
tvec3<T, P> const & v3);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> colMajor3(
tmat3x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> colMajor4(
tvec4<T, P> const & v1, 
tvec4<T, P> const & v2, 
tvec4<T, P> const & v3, 
tvec4<T, P> const & v4);
template <typename T, precision P> 
GLM_FUNC_DECL tmat4x4<T, P> colMajor4(
tmat4x4<T, P> const & m);
}
#include "matrix_major_storage.inl"
