#pragma once
#include "type_vec1.hpp"
#include "type_vec2.hpp"
#include "type_vec3.hpp"
#include "type_vec4.hpp"
#include <cmath>
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> pow(vecType<T, P> const & base, vecType<T, P> const & exponent);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> exp(vecType<T, P> const & v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> log(vecType<T, P> const & v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> exp2(vecType<T, P> const & v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> log2(vecType<T, P> const & v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> sqrt(vecType<T, P> const & v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> inversesqrt(vecType<T, P> const & v);
}
#include "func_exponential.inl"
