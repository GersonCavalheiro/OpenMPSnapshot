
#ifndef GLM_GTX_wrap
#define GLM_GTX_wrap GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_wrap extension included")
#endif

namespace glm
{

template <typename genType> 
genType clamp(genType const & Texcoord);

template <typename genType> 
genType repeat(genType const & Texcoord);

template <typename genType> 
genType mirrorRepeat(genType const & Texcoord);

}

#include "wrap.inl"

#endif
