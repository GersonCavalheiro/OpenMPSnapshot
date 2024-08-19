
#pragma once

#include "../mat3x2.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int3x2 extension included")
#endif

namespace glm
{

typedef mat<3, 2, int, defaultp>	imat3x2;

}
