
#ifndef GLM_GTX_perpendicular
#define GLM_GTX_perpendicular GLM_VERSION

#include "../glm.hpp"
#include "../gtx/projection.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_perpendicular extension included")
#endif

namespace glm
{

template <typename vecType> 
vecType perp(
vecType const & x, 
vecType const & Normal);

}

#include "perpendicular.inl"

#endif
