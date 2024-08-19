
#ifndef GLM_GTX_transform 
#define GLM_GTX_transform GLM_VERSION

#include "../glm.hpp"
#include "../gtc/matrix_transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_transform extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat4x4<T> translate(
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> translate(
detail::tmat4x4<T> const & m, 
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> translate(
detail::tvec3<T> const & v);

template <typename T> 
detail::tmat4x4<T> rotate(
T angle, 
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> rotate(
T angle, 
detail::tvec3<T> const & v);

template <typename T> 
detail::tmat4x4<T> rotate(
detail::tmat4x4<T> const & m, 
T angle, 
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> scale(
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> scale(
detail::tmat4x4<T> const & m, 
T x, T y, T z);

template <typename T> 
detail::tmat4x4<T> scale(
detail::tvec3<T> const & v);

}

#include "transform.inl"

#endif
