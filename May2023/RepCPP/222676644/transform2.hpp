
#ifndef GLM_GTX_transform2
#define GLM_GTX_transform2 GLM_VERSION

#include "../glm.hpp"
#include "../gtx/transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_transform2 extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat3x3<T> shearX2D(
detail::tmat3x3<T> const & m, 
T y);

template <typename T> 
detail::tmat3x3<T> shearY2D(
detail::tmat3x3<T> const & m, 
T x);

template <typename T> 
detail::tmat4x4<T> shearX3D(
const detail::tmat4x4<T> & m,
T y, 
T z);

template <typename T> 
detail::tmat4x4<T> shearY3D(
const detail::tmat4x4<T> & m, 
T x, 
T z);

template <typename T> 
detail::tmat4x4<T> shearZ3D(
const detail::tmat4x4<T> & m, 
T x, 
T y);



template <typename T> 
detail::tmat3x3<T> proj2D(
const detail::tmat3x3<T> & m, 
const detail::tvec3<T>& normal);

template <typename T> 
detail::tmat4x4<T> proj3D(
const detail::tmat4x4<T> & m, 
const detail::tvec3<T>& normal);

template <typename valType> 
detail::tmat4x4<valType> scaleBias(
valType scale, 
valType bias);

template <typename valType> 
detail::tmat4x4<valType> scaleBias(
detail::tmat4x4<valType> const & m, 
valType scale, 
valType bias);

}

#include "transform2.inl"

#endif
