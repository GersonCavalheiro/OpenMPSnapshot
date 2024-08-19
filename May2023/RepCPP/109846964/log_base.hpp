
#ifndef GLM_GTX_log_base
#define GLM_GTX_log_base

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_log_base extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL genType log(
genType const & x, 
genType const & base);

}

#include "log_base.inl"

#endif
