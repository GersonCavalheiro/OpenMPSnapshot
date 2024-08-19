
#ifndef GLM_GTX_rotate_vector
#define GLM_GTX_rotate_vector

#include "../glm.hpp"
#include "../gtx/transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_rotate_vector extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec2<T, P> rotate(
detail::tvec2<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rotate(
detail::tvec3<T, P> const & v,
T const & angle,
detail::tvec3<T, P> const & normal);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> rotate(
detail::tvec4<T, P> const & v,
T const & angle,
detail::tvec3<T, P> const & normal);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rotateX(
detail::tvec3<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rotateY(
detail::tvec3<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rotateZ(
detail::tvec3<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> rotateX(
detail::tvec4<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> rotateY(
detail::tvec4<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> rotateZ(
detail::tvec4<T, P> const & v,
T const & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> orientation(
detail::tvec3<T, P> const & Normal,
detail::tvec3<T, P> const & Up);

}

#include "rotate_vector.inl"

#endif
