#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_log_base extension included")
#endif
namespace glm
{
template <typename genType>
GLM_FUNC_DECL genType log(
genType const & x,
genType const & base);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> sign(
vecType<T, P> const & x,
vecType<T, P> const & base);
}
#include "log_base.inl"
