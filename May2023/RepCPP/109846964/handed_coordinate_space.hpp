
#ifndef GLM_GTX_handed_coordinate_space
#define GLM_GTX_handed_coordinate_space

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_handed_coordinate_space extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL bool rightHanded(
detail::tvec3<T, P> const & tangent,
detail::tvec3<T, P> const & binormal,
detail::tvec3<T, P> const & normal);

template <typename T, precision P>
GLM_FUNC_DECL bool leftHanded(
detail::tvec3<T, P> const & tangent,
detail::tvec3<T, P> const & binormal,
detail::tvec3<T, P> const & normal);

}

#include "handed_coordinate_space.inl"

#endif
