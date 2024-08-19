#pragma once
#include "../gtx/fast_square_root.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_normalize_dot extension included")
#endif
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T normalizeDot(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T fastNormalizeDot(vecType<T, P> const & x, vecType<T, P> const & y);
}
#include "normalize_dot.inl"
