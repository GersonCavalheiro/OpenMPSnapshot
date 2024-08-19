
#pragma once

#include "../ext/vector_uint3.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint3_sized extension included")
#endif

namespace glm
{

typedef vec<3, uint8, defaultp>		u8vec3;

typedef vec<3, uint16, defaultp>	u16vec3;

typedef vec<3, uint32, defaultp>	u32vec3;

typedef vec<3, uint64, defaultp>	u64vec3;

}
