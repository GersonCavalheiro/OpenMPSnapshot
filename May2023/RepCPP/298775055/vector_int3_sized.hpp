
#pragma once

#include "../ext/vector_int3.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_int3_sized extension included")
#endif

namespace glm
{

typedef vec<3, int8, defaultp>		i8vec3;

typedef vec<3, int16, defaultp>		i16vec3;

typedef vec<3, int32, defaultp>		i32vec3;

typedef vec<3, int64, defaultp>		i64vec3;

}
