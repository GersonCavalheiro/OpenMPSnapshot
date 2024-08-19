#pragma once
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_epsilon extension included")
#endif
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> epsilonEqual(
vecType<T, P> const & x,
vecType<T, P> const & y,
T const & epsilon);
template <typename genType>
GLM_FUNC_DECL bool epsilonEqual(
genType const & x,
genType const & y,
genType const & epsilon);
template <typename genType>
GLM_FUNC_DECL typename genType::boolType epsilonNotEqual(
genType const & x,
genType const & y,
typename genType::value_type const & epsilon);
template <typename genType>
GLM_FUNC_DECL bool epsilonNotEqual(
genType const & x,
genType const & y,
genType const & epsilon);
}
#include "epsilon.inl"
