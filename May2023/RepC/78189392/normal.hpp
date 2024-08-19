#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_normal extension included")
#endif
namespace glm
{
template <typename T, precision P> 
GLM_FUNC_DECL tvec3<T, P> triangleNormal(
tvec3<T, P> const & p1, 
tvec3<T, P> const & p2, 
tvec3<T, P> const & p3);
}
#include "normal.inl"
