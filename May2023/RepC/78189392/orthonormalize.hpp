#pragma once
#include "../vec3.hpp"
#include "../mat3x3.hpp"
#include "../geometric.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_orthonormalize extension included")
#endif
namespace glm
{
template <typename T, precision P> 
GLM_FUNC_DECL tmat3x3<T, P> orthonormalize(tmat3x3<T, P> const & m);
template <typename T, precision P> 
GLM_FUNC_DECL tvec3<T, P> orthonormalize(tvec3<T, P> const & x, tvec3<T, P> const & y);
}
#include "orthonormalize.inl"
