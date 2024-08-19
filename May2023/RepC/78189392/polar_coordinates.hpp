#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_polar_coordinates extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> polar(
tvec3<T, P> const & euclidean);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> euclidean(
tvec2<T, P> const & polar);
}
#include "polar_coordinates.inl"
