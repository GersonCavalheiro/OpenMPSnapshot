
#pragma once

#include "../mat2x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint2x4 extension included")
#endif

namespace glm
{

typedef mat<2, 4, uint, defaultp>	umat2x4;

}
