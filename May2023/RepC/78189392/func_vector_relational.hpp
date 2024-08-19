#pragma once
#include "precision.hpp"
#include "setup.hpp"
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> lessThan(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> lessThanEqual(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> greaterThan(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> greaterThanEqual(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> equal(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> notEqual(vecType<T, P> const & x, vecType<T, P> const & y);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool any(vecType<bool, P> const & v);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool all(vecType<bool, P> const & v);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> not_(vecType<bool, P> const & v);
}
#include "func_vector_relational.inl"
