#pragma once
#include "../glm.hpp"
#include "../gtx/quaternion.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_norm extension included")
#endif
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T length2(
vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T distance2(
vecType<T, P> const & p0,
vecType<T, P> const & p1);
template <typename T, precision P>
GLM_FUNC_DECL T l1Norm(
tvec3<T, P> const & x,
tvec3<T, P> const & y);
template <typename T, precision P>
GLM_FUNC_DECL T l1Norm(
tvec3<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL T l2Norm(
tvec3<T, P> const & x,
tvec3<T, P> const & y);
template <typename T, precision P>
GLM_FUNC_DECL T l2Norm(
tvec3<T, P> const & x);
template <typename T, precision P>
GLM_FUNC_DECL T lxNorm(
tvec3<T, P> const & x,
tvec3<T, P> const & y,
unsigned int Depth);
template <typename T, precision P>
GLM_FUNC_DECL T lxNorm(
tvec3<T, P> const & x,
unsigned int Depth);
}
#include "norm.inl"
