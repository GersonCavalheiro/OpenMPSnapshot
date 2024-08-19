
#ifndef GLM_GTX_orthonormalize
#define GLM_GTX_orthonormalize

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_orthonormalize extension included")
#endif

namespace glm
{

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat3x3<T, P> orthonormalize(
const detail::tmat3x3<T, P>& m);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tvec3<T, P> orthonormalize(
const detail::tvec3<T, P>& x, 
const detail::tvec3<T, P>& y);

}

#include "orthonormalize.inl"

#endif
