
#pragma once

#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_extend is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_extend extension included")
#	endif
#endif

namespace glm
{

template<typename genType>
GLM_FUNC_DECL genType extend(
genType const& Origin,
genType const& Source,
typename genType::value_type const Length);

}

#include "extend.inl"
