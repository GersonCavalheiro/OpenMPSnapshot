
#ifndef GLM_GTX_projection
#define GLM_GTX_projection

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_projection extension included")
#endif

namespace glm
{

template <typename vecType> 
GLM_FUNC_DECL vecType proj(
vecType const & x, 
vecType const & Normal);

}

#include "projection.inl"

#endif
