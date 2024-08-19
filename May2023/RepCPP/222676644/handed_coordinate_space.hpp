
#ifndef GLM_GTX_handed_coordinate_space
#define GLM_GTX_handed_coordinate_space GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_handed_coordinate_space extension included")
#endif

namespace glm
{

template <typename T> 
bool rightHanded(
detail::tvec3<T> const & tangent, 
detail::tvec3<T> const & binormal, 
detail::tvec3<T> const & normal);

template <typename T> 
bool leftHanded(
detail::tvec3<T> const & tangent, 
detail::tvec3<T> const & binormal, 
detail::tvec3<T> const & normal);

}

#include "handed_coordinate_space.inl"

#endif
