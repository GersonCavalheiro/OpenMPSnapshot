
#pragma once

#include "../mat3x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int3x4 extension included")
#endif

namespace glm
{

typedef mat<3, 4, int, defaultp>	imat3x4;

}
