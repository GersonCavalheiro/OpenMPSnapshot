#pragma once
#include "../glm.hpp"
#include "../gtx/transform.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_rotate_vector extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> slerp(
tvec3<T, P> const & x,
tvec3<T, P> const & y,
T const & a);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> rotate(
tvec2<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rotate(
tvec3<T, P> const & v,
T const & angle,
tvec3<T, P> const & normal);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> rotate(
tvec4<T, P> const & v,
T const & angle,
tvec3<T, P> const & normal);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rotateX(
tvec3<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rotateY(
tvec3<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rotateZ(
tvec3<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> rotateX(
tvec4<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> rotateY(
tvec4<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> rotateZ(
tvec4<T, P> const & v,
T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> orientation(
tvec3<T, P> const & Normal,
tvec3<T, P> const & Up);
}
#include "rotate_vector.inl"
