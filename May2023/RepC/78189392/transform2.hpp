#pragma once
#include "../glm.hpp"
#include "../gtx/transform.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_transform2 extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> shearX2D(
tmat3x3<T, P> const & m, 
T y);
template <typename T, precision P> 
GLM_FUNC_DECL tmat3x3<T, P> shearY2D(
tmat3x3<T, P> const & m, 
T x);
template <typename T, precision P> 
GLM_FUNC_DECL tmat4x4<T, P> shearX3D(
const tmat4x4<T, P> & m,
T y, 
T z);
template <typename T, precision P> 
GLM_FUNC_DECL tmat4x4<T, P> shearY3D(
const tmat4x4<T, P> & m, 
T x, 
T z);
template <typename T, precision P> 
GLM_FUNC_DECL tmat4x4<T, P> shearZ3D(
const tmat4x4<T, P> & m, 
T x, 
T y);
template <typename T, precision P> 
GLM_FUNC_DECL tmat3x3<T, P> proj2D(
const tmat3x3<T, P> & m, 
const tvec3<T, P>& normal);
template <typename T, precision P> 
GLM_FUNC_DECL tmat4x4<T, P> proj3D(
const tmat4x4<T, P> & m, 
const tvec3<T, P>& normal);
template <typename valType, precision P> 
GLM_FUNC_DECL tmat4x4<valType, P> scaleBias(
valType scale, 
valType bias);
template <typename valType, precision P> 
GLM_FUNC_DECL tmat4x4<valType, P> scaleBias(
tmat4x4<valType, P> const & m, 
valType scale, 
valType bias);
}
#include "transform2.inl"
