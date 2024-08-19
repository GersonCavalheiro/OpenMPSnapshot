
#pragma once

#include "../mat3x3.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int3x3 extension included")
#endif

namespace glm
{

typedef mat<3, 3, int, defaultp>	imat3x3;

typedef mat<3, 3, int, defaultp>	imat3;

}
