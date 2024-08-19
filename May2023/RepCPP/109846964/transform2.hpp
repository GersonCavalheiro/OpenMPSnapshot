
#ifndef GLM_GTX_transform2
#define GLM_GTX_transform2

#include "../glm.hpp"
#include "../gtx/transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_transform2 extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> shearX2D(
detail::tmat3x3<T, P> const & m, 
T y);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat3x3<T, P> shearY2D(
detail::tmat3x3<T, P> const & m, 
T x);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat4x4<T, P> shearX3D(
const detail::tmat4x4<T, P> & m,
T y, 
T z);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat4x4<T, P> shearY3D(
const detail::tmat4x4<T, P> & m, 
T x, 
T z);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat4x4<T, P> shearZ3D(
const detail::tmat4x4<T, P> & m, 
T x, 
T y);



template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat3x3<T, P> proj2D(
const detail::tmat3x3<T, P> & m, 
const detail::tvec3<T, P>& normal);

template <typename T, precision P> 
GLM_FUNC_DECL detail::tmat4x4<T, P> proj3D(
const detail::tmat4x4<T, P> & m, 
const detail::tvec3<T, P>& normal);

template <typename valType, precision P> 
GLM_FUNC_DECL detail::tmat4x4<valType, P> scaleBias(
valType scale, 
valType bias);

template <typename valType, precision P> 
GLM_FUNC_DECL detail::tmat4x4<valType, P> scaleBias(
detail::tmat4x4<valType, P> const & m, 
valType scale, 
valType bias);

}

#include "transform2.inl"

#endif
