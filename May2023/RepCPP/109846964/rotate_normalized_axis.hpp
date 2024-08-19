
#ifndef GLM_GTX_rotate_normalized_axis
#define GLM_GTX_rotate_normalized_axis

#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_rotate_normalized_axis extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> rotateNormalizedAxis(
detail::tmat4x4<T, P> const & m,
T const & angle,
detail::tvec3<T, P> const & axis);

template <typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> rotateNormalizedAxis(
detail::tquat<T, P> const & q,
T const & angle,
detail::tvec3<T, P> const & axis);

}

#include "rotate_normalized_axis.inl"

#endif
