
#pragma once

#include "../ext/vector_int2.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_int2_sized extension included")
#endif

namespace glm
{

typedef vec<2, int8, defaultp>		i8vec2;

typedef vec<2, int16, defaultp>		i16vec2;

typedef vec<2, int32, defaultp>		i32vec2;

typedef vec<2, int64, defaultp>		i64vec2;

}
