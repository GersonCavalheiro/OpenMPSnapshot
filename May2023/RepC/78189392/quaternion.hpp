#pragma once
#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/norm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_quaternion extension included")
#endif
namespace glm
{
template<typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> cross(
tquat<T, P> const & q,
tvec3<T, P> const & v);
template<typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> cross(
tvec3<T, P> const & v,
tquat<T, P> const & q);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> squad(
tquat<T, P> const & q1,
tquat<T, P> const & q2,
tquat<T, P> const & s1,
tquat<T, P> const & s2,
T const & h);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> intermediate(
tquat<T, P> const & prev,
tquat<T, P> const & curr,
tquat<T, P> const & next);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> exp(
tquat<T, P> const & q);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> log(
tquat<T, P> const & q);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> pow(
tquat<T, P> const & x,
T const & y);
template<typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> rotate(
tquat<T, P> const & q,
tvec3<T, P> const & v);
template<typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> rotate(
tquat<T, P> const & q,
tvec4<T, P> const & v);
template<typename T, precision P>
GLM_FUNC_DECL T extractRealComponent(
tquat<T, P> const & q);
template<typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> toMat3(
tquat<T, P> const & x){return mat3_cast(x);}
template<typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> toMat4(
tquat<T, P> const & x){return mat4_cast(x);}
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> toQuat(
tmat3x3<T, P> const & x){return quat_cast(x);}
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> toQuat(
tmat4x4<T, P> const & x){return quat_cast(x);}
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> shortMix(
tquat<T, P> const & x,
tquat<T, P> const & y,
T const & a);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> fastMix(
tquat<T, P> const & x,
tquat<T, P> const & y,
T const & a);
template<typename T, precision P>
GLM_FUNC_DECL tquat<T, P> rotation(
tvec3<T, P> const & orig, 
tvec3<T, P> const & dest);
template<typename T, precision P>
GLM_FUNC_DECL T length2(tquat<T, P> const & q);
}
#include "quaternion.inl"
