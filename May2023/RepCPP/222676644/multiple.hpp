
#ifndef GLM_GTX_multiple
#define GLM_GTX_multiple GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_multiple extension included")
#endif

namespace glm
{

template <typename genType> 
genType higherMultiple(
genType const & Source, 
genType const & Multiple);

template <typename genType> 
genType lowerMultiple(
genType const & Source, 
genType const & Multiple);

}

#include "multiple.inl"

#endif
