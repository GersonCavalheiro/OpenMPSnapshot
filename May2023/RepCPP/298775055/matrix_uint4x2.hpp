
#pragma once

#include "../mat4x2.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint4x2 extension included")
#endif

namespace glm
{

typedef mat<4, 2, uint, defaultp>	umat4x2;

}
