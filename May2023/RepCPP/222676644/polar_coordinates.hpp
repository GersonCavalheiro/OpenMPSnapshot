
#ifndef GLM_GTX_polar_coordinates
#define GLM_GTX_polar_coordinates GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_polar_coordinates extension included")
#endif

namespace glm
{

template <typename T> 
detail::tvec3<T> polar(
detail::tvec3<T> const & euclidean);

template <typename T> 
detail::tvec3<T> euclidean(
detail::tvec2<T> const & polar);

}

#include "polar_coordinates.inl"

#endif
