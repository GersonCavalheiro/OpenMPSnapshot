
#pragma once

#include "../mat2x3.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint2x3 extension included")
#endif

namespace glm
{

typedef mat<2, 3, uint, defaultp>	umat2x3;

}
