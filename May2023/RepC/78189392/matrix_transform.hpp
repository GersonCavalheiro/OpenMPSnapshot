#pragma once
#include "../mat4x4.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../gtc/constants.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_matrix_transform extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> translate(
tmat4x4<T, P> const & m,
tvec3<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> rotate(
tmat4x4<T, P> const & m,
T angle,
tvec3<T, P> const & axis);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> scale(
tmat4x4<T, P> const & m,
tvec3<T, P> const & v);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> ortho(
T left,
T right,
T bottom,
T top,
T zNear,
T zFar);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> ortho(
T left,
T right,
T bottom,
T top);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> frustum(
T left,
T right,
T bottom,
T top,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspective(
T fovy,
T aspect,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspectiveRH(
T fovy,
T aspect,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspectiveLH(
T fovy,
T aspect,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspectiveFov(
T fov,
T width,
T height,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspectiveFovRH(
T fov,
T width,
T height,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> perspectiveFovLH(
T fov,
T width,
T height,
T near,
T far);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> infinitePerspective(
T fovy, T aspect, T near);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> tweakedInfinitePerspective(
T fovy, T aspect, T near);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> tweakedInfinitePerspective(
T fovy, T aspect, T near, T ep);
template <typename T, typename U, precision P>
GLM_FUNC_DECL tvec3<T, P> project(
tvec3<T, P> const & obj,
tmat4x4<T, P> const & model,
tmat4x4<T, P> const & proj,
tvec4<U, P> const & viewport);
template <typename T, typename U, precision P>
GLM_FUNC_DECL tvec3<T, P> unProject(
tvec3<T, P> const & win,
tmat4x4<T, P> const & model,
tmat4x4<T, P> const & proj,
tvec4<U, P> const & viewport);
template <typename T, precision P, typename U>
GLM_FUNC_DECL tmat4x4<T, P> pickMatrix(
tvec2<T, P> const & center,
tvec2<T, P> const & delta,
tvec4<U, P> const & viewport);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> lookAt(
tvec3<T, P> const & eye,
tvec3<T, P> const & center,
tvec3<T, P> const & up);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> lookAtRH(
tvec3<T, P> const & eye,
tvec3<T, P> const & center,
tvec3<T, P> const & up);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> lookAtLH(
tvec3<T, P> const & eye,
tvec3<T, P> const & center,
tvec3<T, P> const & up);
}
#include "matrix_transform.inl"
