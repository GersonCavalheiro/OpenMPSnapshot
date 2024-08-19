
#pragma once

#include "../mat4x4.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int4x4_sized extension included")
#endif

namespace glm
{

typedef mat<4, 4, int8, defaultp>				i8mat4x4;

typedef mat<4, 4, int16, defaultp>				i16mat4x4;

typedef mat<4, 4, int32, defaultp>				i32mat4x4;

typedef mat<4, 4, int64, defaultp>				i64mat4x4;


typedef mat<4, 4, int8, defaultp>				i8mat4;

typedef mat<4, 4, int16, defaultp>				i16mat4;

typedef mat<4, 4, int32, defaultp>				i32mat4;

typedef mat<4, 4, int64, defaultp>				i64mat4;

}
