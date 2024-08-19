
#ifndef GLM_GTX_transform 
#define GLM_GTX_transform

#include "../glm.hpp"
#include "../gtc/matrix_transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_transform extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> translate(
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> rotate(
T angle, 
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> scale(
detail::tvec3<T, P> const & v);

}

#include "transform.inl"

#endif
