
#ifndef GLM_GTX_polar_coordinates
#define GLM_GTX_polar_coordinates

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_polar_coordinates extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> polar(
detail::tvec3<T, P> const & euclidean);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> euclidean(
detail::tvec2<T, P> const & polar);

}

#include "polar_coordinates.inl"

#endif
