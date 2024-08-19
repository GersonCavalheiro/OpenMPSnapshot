
#ifndef GLM_GTX_quaternion
#define GLM_GTX_quaternion

#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/norm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_quaternion extension included")
#endif

namespace glm
{

template<typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> cross(
detail::tquat<T, P> const & q,
detail::tvec3<T, P> const & v);

template<typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> cross(
detail::tvec3<T, P> const & v,
detail::tquat<T, P> const & q);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> squad(
detail::tquat<T, P> const & q1,
detail::tquat<T, P> const & q2,
detail::tquat<T, P> const & s1,
detail::tquat<T, P> const & s2,
T const & h);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> intermediate(
detail::tquat<T, P> const & prev,
detail::tquat<T, P> const & curr,
detail::tquat<T, P> const & next);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> exp(
detail::tquat<T, P> const & q);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> log(
detail::tquat<T, P> const & q);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> pow(
detail::tquat<T, P> const & x,
T const & y);


template<typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rotate(
detail::tquat<T, P> const & q,
detail::tvec3<T, P> const & v);

template<typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> rotate(
detail::tquat<T, P> const & q,
detail::tvec4<T, P> const & v);

template<typename T, precision P>
GLM_FUNC_DECL T extractRealComponent(
detail::tquat<T, P> const & q);

template<typename T, precision P>
GLM_FUNC_DECL detail::tmat3x3<T, P> toMat3(
detail::tquat<T, P> const & x){return mat3_cast(x);}

template<typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> toMat4(
detail::tquat<T, P> const & x){return mat4_cast(x);}

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> toQuat(
detail::tmat3x3<T, P> const & x){return quat_cast(x);}

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> toQuat(
detail::tmat4x4<T, P> const & x){return quat_cast(x);}

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> shortMix(
detail::tquat<T, P> const & x,
detail::tquat<T, P> const & y,
T const & a);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> fastMix(
detail::tquat<T, P> const & x,
detail::tquat<T, P> const & y,
T const & a);

template<typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> rotation(
detail::tvec3<T, P> const & orig, 
detail::tvec3<T, P> const & dest);

template<typename T, precision P>
GLM_FUNC_DECL T length2(detail::tquat<T, P> const & q);

}

#include "quaternion.inl"

#endif
