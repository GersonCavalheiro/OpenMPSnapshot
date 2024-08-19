
#ifndef GLM_GTX_normal
#define GLM_GTX_normal GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_normal extension included")
#endif

namespace glm
{

template <typename T> 
detail::tvec3<T> triangleNormal(
detail::tvec3<T> const & p1, 
detail::tvec3<T> const & p2, 
detail::tvec3<T> const & p3);

}

#include "normal.inl"

#endif
