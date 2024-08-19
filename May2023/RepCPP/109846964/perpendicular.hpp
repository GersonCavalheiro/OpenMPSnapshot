
#ifndef GLM_GTX_perpendicular
#define GLM_GTX_perpendicular

#include "../glm.hpp"
#include "../gtx/projection.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_perpendicular extension included")
#endif

namespace glm
{

template <typename vecType> 
GLM_FUNC_DECL vecType perp(
vecType const & x, 
vecType const & Normal);

}

#include "perpendicular.inl"

#endif
