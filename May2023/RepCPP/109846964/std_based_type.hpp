
#ifndef GLM_GTX_std_based_type
#define GLM_GTX_std_based_type

#include "../glm.hpp"
#include <cstdlib>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_std_based_type extension included")
#endif

namespace glm
{

typedef detail::tvec2<std::size_t, defaultp>		size2;

typedef detail::tvec3<std::size_t, defaultp>		size3;

typedef detail::tvec4<std::size_t, defaultp>		size4;

typedef detail::tvec2<std::size_t, defaultp>		size2_t;

typedef detail::tvec3<std::size_t, defaultp>		size3_t;

typedef detail::tvec4<std::size_t, defaultp>		size4_t;

}

#include "std_based_type.inl"

#endif
