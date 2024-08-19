
#pragma once

#include "../ext/vector_uint2.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint2_sized extension included")
#endif

namespace glm
{

typedef vec<2, uint8, defaultp>		u8vec2;

typedef vec<2, uint16, defaultp>	u16vec2;

typedef vec<2, uint32, defaultp>	u32vec2;

typedef vec<2, uint64, defaultp>	u64vec2;

}
