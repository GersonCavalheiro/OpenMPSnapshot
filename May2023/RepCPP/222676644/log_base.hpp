
#ifndef GLM_GTX_log_base
#define GLM_GTX_log_base GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_log_base extension included")
#endif

namespace glm
{

template <typename genType> 
genType log(
genType const & x, 
genType const & base);

}

#include "log_base.inl"

#endif
