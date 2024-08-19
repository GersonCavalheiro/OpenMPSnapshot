#pragma once
#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtx/quaternion.hpp"
#include "../gtx/rotate_vector.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_vector_angle extension included")
#endif
namespace glm
{
template <typename vecType>
GLM_FUNC_DECL typename vecType::value_type angle(
vecType const & x, 
vecType const & y);
template <typename T, precision P>
GLM_FUNC_DECL T orientedAngle(
tvec2<T, P> const & x,
tvec2<T, P> const & y);
template <typename T, precision P>
GLM_FUNC_DECL T orientedAngle(
tvec3<T, P> const & x,
tvec3<T, P> const & y,
tvec3<T, P> const & ref);
}
#include "vector_angle.inl"
