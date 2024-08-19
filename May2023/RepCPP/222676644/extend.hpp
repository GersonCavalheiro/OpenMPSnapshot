
#ifndef GLM_GTX_extend
#define GLM_GTX_extend GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_extend extension included")
#endif

namespace glm
{

template <typename genType> 
genType extend(
genType const & Origin, 
genType const & Source, 
typename genType::value_type const Length);

}

#include "extend.inl"

#endif
