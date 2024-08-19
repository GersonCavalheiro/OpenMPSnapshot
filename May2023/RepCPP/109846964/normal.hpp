
#ifndef GLM_GTX_normal
#define GLM_GTX_normal

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_normal extension included")
#endif

namespace glm
{

template <typename T, precision P> 
GLM_FUNC_DECL detail::tvec3<T, P> triangleNormal(
detail::tvec3<T, P> const & p1, 
detail::tvec3<T, P> const & p2, 
detail::tvec3<T, P> const & p3);

}

#include "normal.inl"

#endif
