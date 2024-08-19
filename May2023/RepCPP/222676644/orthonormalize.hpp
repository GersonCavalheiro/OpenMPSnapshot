
#ifndef GLM_GTX_orthonormalize
#define GLM_GTX_orthonormalize GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_orthonormalize extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat3x3<T> orthonormalize(
const detail::tmat3x3<T>& m);

template <typename T> 
detail::tvec3<T> orthonormalize(
const detail::tvec3<T>& x, 
const detail::tvec3<T>& y);

}

#include "orthonormalize.inl"

#endif
