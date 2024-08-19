
#ifndef GLM_GTC_matrix_transform
#define GLM_GTC_matrix_transform

#include "../mat4x4.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../gtc/constants.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_matrix_transform extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> translate(
detail::tmat4x4<T, P> const & m,
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> rotate(
detail::tmat4x4<T, P> const & m,
T const & angle,
detail::tvec3<T, P> const & axis);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> scale(
detail::tmat4x4<T, P> const & m,
detail::tvec3<T, P> const & v);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> ortho(
T const & left,
T const & right,
T const & bottom,
T const & top,
T const & zNear,
T const & zFar);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> ortho(
T const & left,
T const & right,
T const & bottom,
T const & top);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> frustum(
T const & left,
T const & right,
T const & bottom,
T const & top,
T const & near,
T const & far);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> perspective(
T const & fovy,
T const & aspect,
T const & near,
T const & far);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> perspectiveFov(
T const & fov,
T const & width,
T const & height,
T const & near,
T const & far);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> infinitePerspective(
T fovy, T aspect, T near);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> tweakedInfinitePerspective(
T fovy, T aspect, T near);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> tweakedInfinitePerspective(
T fovy, T aspect, T near, T ep);

template <typename T, typename U, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> project(
detail::tvec3<T, P> const & obj,
detail::tmat4x4<T, P> const & model,
detail::tmat4x4<T, P> const & proj,
detail::tvec4<U, P> const & viewport);

template <typename T, typename U, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> unProject(
detail::tvec3<T, P> const & win,
detail::tmat4x4<T, P> const & model,
detail::tmat4x4<T, P> const & proj,
detail::tvec4<U, P> const & viewport);

template <typename T, precision P, typename U>
GLM_FUNC_DECL detail::tmat4x4<T, P> pickMatrix(
detail::tvec2<T, P> const & center,
detail::tvec2<T, P> const & delta,
detail::tvec4<U, P> const & viewport);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> lookAt(
detail::tvec3<T, P> const & eye,
detail::tvec3<T, P> const & center,
detail::tvec3<T, P> const & up);

}

#include "matrix_transform.inl"

#endif
