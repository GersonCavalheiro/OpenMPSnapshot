#pragma once
#include "../glm.hpp"
#include <cstdlib>
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_std_based_type extension included")
#endif
namespace glm
{
typedef tvec1<std::size_t, defaultp>		size1;
typedef tvec2<std::size_t, defaultp>		size2;
typedef tvec3<std::size_t, defaultp>		size3;
typedef tvec4<std::size_t, defaultp>		size4;
typedef tvec1<std::size_t, defaultp>		size1_t;
typedef tvec2<std::size_t, defaultp>		size2_t;
typedef tvec3<std::size_t, defaultp>		size3_t;
typedef tvec4<std::size_t, defaultp>		size4_t;
}
#include "std_based_type.inl"
