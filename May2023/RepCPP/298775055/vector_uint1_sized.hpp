
#pragma once

#include "../ext/vector_uint1.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_uint1_sized extension included")
#endif

namespace glm
{

typedef vec<1, uint8, defaultp>		u8vec1;

typedef vec<1, uint16, defaultp>	u16vec1;

typedef vec<1, uint32, defaultp>	u32vec1;

typedef vec<1, uint64, defaultp>	u64vec1;

}
