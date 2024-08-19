#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_optimum_pow extension included")
#endif
namespace glm{
namespace gtx
{
template <typename genType>
GLM_FUNC_DECL genType pow2(genType const & x);
template <typename genType>
GLM_FUNC_DECL genType pow3(genType const & x);
template <typename genType>
GLM_FUNC_DECL genType pow4(genType const & x);
}
}
#include "optimum_pow.inl"
