#pragma once
#include "setup.hpp"
#include "precision.hpp"
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL GLM_CONSTEXPR vecType<T, P> radians(vecType<T, P> const & degrees);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL GLM_CONSTEXPR vecType<T, P> degrees(vecType<T, P> const & radians);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> sin(vecType<T, P> const & angle);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> cos(vecType<T, P> const & angle);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> tan(vecType<T, P> const & angle); 
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> asin(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> acos(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> atan(vecType<T, P> const & y, vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> atan(vecType<T, P> const & y_over_x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> sinh(vecType<T, P> const & angle);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> cosh(vecType<T, P> const & angle);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> tanh(vecType<T, P> const & angle);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> asinh(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> acosh(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> atanh(vecType<T, P> const & x);
}
#include "func_trigonometric.inl"
