
#pragma once

#include "../glm.hpp"
#include "../ext/scalar_common.hpp"
#include "../ext/vector_common.hpp"
#include "../gtc/vec1.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_wrap is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_wrap extension included")
#	endif
#endif

namespace glm
{

}

#include "wrap.inl"
