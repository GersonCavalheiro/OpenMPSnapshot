
#pragma once

#include "../ext/vector_int1.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_int1_sized extension included")
#endif

namespace glm
{

typedef vec<1, int8, defaultp>	i8vec1;

typedef vec<1, int16, defaultp>	i16vec1;

typedef vec<1, int32, defaultp>	i32vec1;

typedef vec<1, int64, defaultp>	i64vec1;

}
