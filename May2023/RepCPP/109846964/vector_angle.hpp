
#ifndef GLM_GTX_vector_angle
#define GLM_GTX_vector_angle

#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtx/quaternion.hpp"
#include "../gtx/rotate_vector.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_vector_angle extension included")
#endif

namespace glm
{

template <typename vecType>
GLM_FUNC_DECL typename vecType::value_type angle(
vecType const & x, 
vecType const & y);

template <typename T, precision P>
GLM_FUNC_DECL T orientedAngle(
detail::tvec2<T, P> const & x,
detail::tvec2<T, P> const & y);

template <typename T, precision P>
GLM_FUNC_DECL T orientedAngle(
detail::tvec3<T, P> const & x,
detail::tvec3<T, P> const & y,
detail::tvec3<T, P> const & ref);

}

#include "vector_angle.inl"

#endif
