
#ifndef GLM_GTX_vector_angle
#define GLM_GTX_vector_angle GLM_VERSION

#include "../glm.hpp"
#include "../gtc/epsilon.hpp"
#include "../gtx/quaternion.hpp"
#include "../gtx/rotate_vector.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_vector_angle extension included")
#endif

namespace glm
{

template <typename vecType> 
GLM_FUNC_QUALIFIER typename vecType::value_type angle(
vecType const & x, 
vecType const & y);

template <typename T> 
GLM_FUNC_QUALIFIER T orientedAngle(
detail::tvec2<T> const & x, 
detail::tvec2<T> const & y);

template <typename T>
GLM_FUNC_QUALIFIER T orientedAngle(
detail::tvec3<T> const & x,
detail::tvec3<T> const & y,
detail::tvec3<T> const & ref);

}

#include "vector_angle.inl"

#endif
