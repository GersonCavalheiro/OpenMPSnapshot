
#ifndef GLM_GTX_wrap
#define GLM_GTX_wrap

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_wrap extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL genType clamp(genType const & Texcoord);

template <typename genType> 
GLM_FUNC_DECL genType repeat(genType const & Texcoord);

template <typename genType> 
GLM_FUNC_DECL genType mirrorRepeat(genType const & Texcoord);

}

#include "wrap.inl"

#endif
