
#ifndef GLM_GTX_norm
#define GLM_GTX_norm GLM_VERSION

#include "../glm.hpp"
#include "../gtx/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_norm extension included")
#endif

namespace glm
{

template <typename T> 
T length2(
T const & x);

template <typename genType> 
typename genType::value_type length2(
genType const & x);

template <typename T>
T length2(
detail::tquat<T> const & q);

template <typename T>
T distance2(
T const & p0, 
T const & p1);

template <typename genType> 
typename genType::value_type distance2(
genType const & p0, 
genType const & p1);

template <typename T>
T l1Norm(
detail::tvec3<T> const & x,
detail::tvec3<T> const & y);

template <typename T> 
T l1Norm(
detail::tvec3<T> const & v);

template <typename T> 
T l2Norm(
detail::tvec3<T> const & x, 
detail::tvec3<T> const & y);

template <typename T> 
T l2Norm(
detail::tvec3<T> const & x);

template <typename T> 
T lxNorm(
detail::tvec3<T> const & x,
detail::tvec3<T> const & y,
unsigned int Depth);

template <typename T>
T lxNorm(
detail::tvec3<T> const & x,
unsigned int Depth);

}

#include "norm.inl"

#endif
