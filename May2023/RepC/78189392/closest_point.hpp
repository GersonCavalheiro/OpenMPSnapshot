#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_closest_point extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> closestPointOnLine(
tvec3<T, P> const & point,
tvec3<T, P> const & a, 
tvec3<T, P> const & b);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> closestPointOnLine(
tvec2<T, P> const & point,
tvec2<T, P> const & a, 
tvec2<T, P> const & b);	
}
#include "closest_point.inl"
