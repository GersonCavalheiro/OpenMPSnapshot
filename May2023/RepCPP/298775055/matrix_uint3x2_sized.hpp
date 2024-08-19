
#pragma once

#include "../mat3x2.hpp"
#include "../ext/scalar_uint_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint3x2_sized extension included")
#endif

namespace glm
{

typedef mat<3, 2, uint8, defaultp>				u8mat3x2;

typedef mat<3, 2, uint16, defaultp>				u16mat3x2;

typedef mat<3, 2, uint32, defaultp>				u32mat3x2;

typedef mat<3, 2, uint64, defaultp>				u64mat3x2;

}
