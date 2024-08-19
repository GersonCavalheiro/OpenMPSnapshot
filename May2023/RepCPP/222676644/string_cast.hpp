
#ifndef GLM_GTX_string_cast
#define GLM_GTX_string_cast GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"
#include "../gtx/integer.hpp"
#include "../gtx/quaternion.hpp"
#include <string>

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_string_cast extension included")
#endif

namespace glm
{

template <typename genType> 
std::string to_string(genType const & x);

}

#include "string_cast.inl"

#endif
