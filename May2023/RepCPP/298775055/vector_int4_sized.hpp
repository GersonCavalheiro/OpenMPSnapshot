
#pragma once

#include "../ext/vector_int4.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_int4_sized extension included")
#endif

namespace glm
{

typedef vec<4, int8, defaultp>		i8vec4;

typedef vec<4, int16, defaultp>		i16vec4;

typedef vec<4, int32, defaultp>		i32vec4;

typedef vec<4, int64, defaultp>		i64vec4;

}
