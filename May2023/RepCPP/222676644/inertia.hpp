
#ifndef GLM_GTX_inertia
#define GLM_GTX_inertia GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_inertia extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat3x3<T> boxInertia3(
T const & Mass, 
detail::tvec3<T> const & Scale);

template <typename T> 
detail::tmat4x4<T> boxInertia4(
T const & Mass, 
detail::tvec3<T> const & Scale);

template <typename T> 
detail::tmat3x3<T> diskInertia3(
T const & Mass, 
T const & Radius);

template <typename T> 
detail::tmat4x4<T> diskInertia4(
T const & Mass, 
T const & Radius);

template <typename T> 
detail::tmat3x3<T> ballInertia3(
T const & Mass, 
T const & Radius);

template <typename T> 
detail::tmat4x4<T> ballInertia4(
T const & Mass, 
T const & Radius);

template <typename T> 
detail::tmat3x3<T> sphereInertia3(
T const & Mass, 
T const & Radius);

template <typename T> 
detail::tmat4x4<T> sphereInertia4(
T const & Mass, 
T const & Radius);

}

#include "inertia.inl"

#endif
