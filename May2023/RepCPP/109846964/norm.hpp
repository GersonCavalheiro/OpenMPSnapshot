
#ifndef GLM_GTX_norm
#define GLM_GTX_norm

#include "../glm.hpp"
#include "../gtx/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_norm extension included")
#endif

namespace glm
{

template <typename T>
GLM_FUNC_DECL T length2(
T const & x);

template <typename genType>
GLM_FUNC_DECL typename genType::value_type length2(
genType const & x);

template <typename T>
GLM_FUNC_DECL T distance2(
T const & p0,
T const & p1);

template <typename genType>
GLM_FUNC_DECL typename genType::value_type distance2(
genType const & p0,
genType const & p1);

template <typename T, precision P>
GLM_FUNC_DECL T l1Norm(
detail::tvec3<T, P> const & x,
detail::tvec3<T, P> const & y);

template <typename T, precision P>
GLM_FUNC_DECL T l1Norm(
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL T l2Norm(
detail::tvec3<T, P> const & x,
detail::tvec3<T, P> const & y);

template <typename T, precision P>
GLM_FUNC_DECL T l2Norm(
detail::tvec3<T, P> const & x);

template <typename T, precision P>
GLM_FUNC_DECL T lxNorm(
detail::tvec3<T, P> const & x,
detail::tvec3<T, P> const & y,
unsigned int Depth);

template <typename T, precision P>
GLM_FUNC_DECL T lxNorm(
detail::tvec3<T, P> const & x,
unsigned int Depth);

}

#include "norm.inl"

#endif
