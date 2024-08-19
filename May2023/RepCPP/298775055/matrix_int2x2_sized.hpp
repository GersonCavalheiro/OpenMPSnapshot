
#pragma once

#include "../mat2x2.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int2x2_sized extension included")
#endif

namespace glm
{

typedef mat<2, 2, int8, defaultp>				i8mat2x2;

typedef mat<2, 2, int16, defaultp>				i16mat2x2;

typedef mat<2, 2, int32, defaultp>				i32mat2x2;

typedef mat<2, 2, int64, defaultp>				i64mat2x2;


typedef mat<2, 2, int8, defaultp>				i8mat2;

typedef mat<2, 2, int16, defaultp>				i16mat2;

typedef mat<2, 2, int32, defaultp>				i32mat2;

typedef mat<2, 2, int64, defaultp>				i64mat2;

}
