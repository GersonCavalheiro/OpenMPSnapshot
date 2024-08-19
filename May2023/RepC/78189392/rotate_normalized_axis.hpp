#pragma once
#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtc/quaternion.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_rotate_normalized_axis extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> rotateNormalizedAxis(
tmat4x4<T, P> const & m,
T const & angle,
tvec3<T, P> const & axis);
template <typename T, precision P>
GLM_FUNC_DECL tquat<T, P> rotateNormalizedAxis(
tquat<T, P> const & q,
T const & angle,
tvec3<T, P> const & axis);
}
#include "rotate_normalized_axis.inl"
