
#ifndef GLM_GTX_quaternion
#define GLM_GTX_quaternion GLM_VERSION

#include "../glm.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_quaternion extension included")
#endif

namespace glm
{

template <typename valType> 
detail::tvec3<valType> cross(
detail::tquat<valType> const & q, 
detail::tvec3<valType> const & v);

template <typename valType> 
detail::tvec3<valType> cross(
detail::tvec3<valType> const & v, 
detail::tquat<valType> const & q);

template <typename valType> 
detail::tquat<valType> squad(
detail::tquat<valType> const & q1, 
detail::tquat<valType> const & q2, 
detail::tquat<valType> const & s1, 
detail::tquat<valType> const & s2, 
valType const & h);

template <typename valType> 
detail::tquat<valType> intermediate(
detail::tquat<valType> const & prev, 
detail::tquat<valType> const & curr, 
detail::tquat<valType> const & next);

template <typename valType> 
detail::tquat<valType> exp(
detail::tquat<valType> const & q);

template <typename valType> 
detail::tquat<valType> log(
detail::tquat<valType> const & q);

template <typename valType> 
detail::tquat<valType> pow(
detail::tquat<valType> const & x, 
valType const & y);


template <typename valType> 
detail::tvec3<valType> rotate(
detail::tquat<valType> const & q, 
detail::tvec3<valType> const & v);

template <typename valType> 
detail::tvec4<valType> rotate(
detail::tquat<valType> const & q, 
detail::tvec4<valType> const & v);

template <typename valType> 
valType extractRealComponent(
detail::tquat<valType> const & q);

template <typename valType> 
detail::tmat3x3<valType> toMat3(
detail::tquat<valType> const & x){return mat3_cast(x);}

template <typename valType> 
detail::tmat4x4<valType> toMat4(
detail::tquat<valType> const & x){return mat4_cast(x);}

template <typename valType> 
detail::tquat<valType> toQuat(
detail::tmat3x3<valType> const & x){return quat_cast(x);}

template <typename valType> 
detail::tquat<valType> toQuat(
detail::tmat4x4<valType> const & x){return quat_cast(x);}

template <typename T>
detail::tquat<T> shortMix(
detail::tquat<T> const & x, 
detail::tquat<T> const & y, 
T const & a);

template <typename T>
detail::tquat<T> fastMix(
detail::tquat<T> const & x, 
detail::tquat<T> const & y, 
T const & a);

}

#include "quaternion.inl"

#endif
