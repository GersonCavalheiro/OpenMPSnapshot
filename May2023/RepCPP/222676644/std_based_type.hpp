
#ifndef GLM_GTX_std_based_type
#define GLM_GTX_std_based_type GLM_VERSION

#include "../glm.hpp"
#include <cstdlib>

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_std_based_type extension included")
#endif

namespace glm
{

typedef detail::tvec2<std::size_t>		size2;

typedef detail::tvec3<std::size_t>		size3;

typedef detail::tvec4<std::size_t>		size4;

typedef detail::tvec2<std::size_t>		size2_t;

typedef detail::tvec3<std::size_t>		size3_t;

typedef detail::tvec4<std::size_t>		size4_t;

}

#include "std_based_type.inl"

#endif
