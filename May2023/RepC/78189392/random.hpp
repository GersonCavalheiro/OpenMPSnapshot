#pragma once
#include "../vec2.hpp"
#include "../vec3.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_random extension included")
#endif
namespace glm
{
template <typename genTYpe>
GLM_FUNC_DECL genTYpe linearRand(
genTYpe Min,
genTYpe Max);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> linearRand(
vecType<T, P> const & Min,
vecType<T, P> const & Max);
template <typename genType>
GLM_FUNC_DECL genType gaussRand(
genType Mean,
genType Deviation);
template <typename T>
GLM_FUNC_DECL tvec2<T, defaultp> circularRand(
T Radius);
template <typename T>
GLM_FUNC_DECL tvec3<T, defaultp> sphericalRand(
T Radius);
template <typename T>
GLM_FUNC_DECL tvec2<T, defaultp> diskRand(
T Radius);
template <typename T>
GLM_FUNC_DECL tvec3<T, defaultp> ballRand(
T Radius);
}
#include "random.inl"
