
#ifndef GLM_GTX_projection
#define GLM_GTX_projection GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_projection extension included")
#endif

namespace glm
{

template <typename vecType> 
vecType proj(
vecType const & x, 
vecType const & Normal);

}

#include "projection.inl"

#endif
