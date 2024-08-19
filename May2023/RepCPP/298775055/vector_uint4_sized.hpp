
#pragma once

#include "../ext/vector_uint4.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint4_sized extension included")
#endif

namespace glm
{

typedef vec<4, uint8, defaultp>		u8vec4;

typedef vec<4, uint16, defaultp>	u16vec4;

typedef vec<4, uint32, defaultp>	u32vec4;

typedef vec<4, uint64, defaultp>	u64vec4;

}
