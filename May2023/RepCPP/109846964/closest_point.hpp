
#ifndef GLM_GTX_closest_point
#define GLM_GTX_closest_point

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> closestPointOnLine(
detail::tvec3<T, P> const & point,
detail::tvec3<T, P> const & a, 
detail::tvec3<T, P> const & b);

}

#include "closest_point.inl"

#endif
