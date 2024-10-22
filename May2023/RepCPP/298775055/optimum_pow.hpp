
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_optimum_pow is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_optimum_pow extension included")
#	endif
#endif

namespace glm{
namespace gtx
{

template<typename genType>
GLM_FUNC_DECL genType pow2(genType const& x);

template<typename genType>
GLM_FUNC_DECL genType pow3(genType const& x);

template<typename genType>
GLM_FUNC_DECL genType pow4(genType const& x);

}
}

#include "optimum_pow.inl"
