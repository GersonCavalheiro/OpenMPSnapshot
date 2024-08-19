
#pragma once

#include "../mat4x2.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int4x2_sized extension included")
#endif

namespace glm
{

typedef mat<4, 2, int8, defaultp>				i8mat4x2;

typedef mat<4, 2, int16, defaultp>				i16mat4x2;

typedef mat<4, 2, int32, defaultp>				i32mat4x2;

typedef mat<4, 2, int64, defaultp>				i64mat4x2;

}
