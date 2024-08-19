
#ifndef GLM_GTC_matrix_transform
#define GLM_GTC_matrix_transform GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_matrix_transform extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat4x4<T> translate(
detail::tmat4x4<T> const & m,
detail::tvec3<T> const & v);

template <typename T> 
detail::tmat4x4<T> rotate(
detail::tmat4x4<T> const & m,
T const & angle, 
detail::tvec3<T> const & axis);

template <typename T> 
detail::tmat4x4<T> scale(
detail::tmat4x4<T> const & m,
detail::tvec3<T> const & v);

template <typename T> 
detail::tmat4x4<T> ortho(
T const & left, 
T const & right, 
T const & bottom, 
T const & top, 
T const & zNear, 
T const & zFar);

template <typename T> 
detail::tmat4x4<T> ortho(
T const & left, 
T const & right, 
T const & bottom, 
T const & top);

template <typename T> 
detail::tmat4x4<T> frustum(
T const & left, 
T const & right, 
T const & bottom, 
T const & top, 
T const & near, 
T const & far);

template <typename T> 
detail::tmat4x4<T> perspective(
T const & fovy, 
T const & aspect, 
T const & near, 
T const & far);

template <typename valType> 
detail::tmat4x4<valType> perspectiveFov(
valType const & fov, 
valType const & width, 
valType const & height, 
valType const & near, 
valType const & far);

template <typename T> 
detail::tmat4x4<T> infinitePerspective(
T fovy, T aspect, T near);

template <typename T> 
detail::tmat4x4<T> tweakedInfinitePerspective(
T fovy, T aspect, T near);

template <typename T, typename U> 
detail::tvec3<T> project(
detail::tvec3<T> const & obj, 
detail::tmat4x4<T> const & model, 
detail::tmat4x4<T> const & proj, 
detail::tvec4<U> const & viewport);

template <typename T, typename U> 
detail::tvec3<T> unProject(
detail::tvec3<T> const & win, 
detail::tmat4x4<T> const & model, 
detail::tmat4x4<T> const & proj, 
detail::tvec4<U> const & viewport);

template <typename T, typename U> 
detail::tmat4x4<T> pickMatrix(
detail::tvec2<T> const & center, 
detail::tvec2<T> const & delta, 
detail::tvec4<U> const & viewport);

template <typename T> 
detail::tmat4x4<T> lookAt(
detail::tvec3<T> const & eye, 
detail::tvec3<T> const & center, 
detail::tvec3<T> const & up);

}

#include "matrix_transform.inl"

#endif
