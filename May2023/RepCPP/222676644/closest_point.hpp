
#ifndef GLM_GTX_closest_point
#define GLM_GTX_closest_point GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{

template <typename T> 
detail::tvec3<T> closestPointOnLine(
detail::tvec3<T> const & point, 
detail::tvec3<T> const & a, 
detail::tvec3<T> const & b);

}

#include "closest_point.inl"

#endif
