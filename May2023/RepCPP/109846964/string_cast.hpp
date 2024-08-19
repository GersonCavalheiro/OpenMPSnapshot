
#ifndef GLM_GTX_string_cast
#define GLM_GTX_string_cast

#include "../glm.hpp"
#include "../gtx/integer.hpp"
#include "../gtx/quaternion.hpp"
#include <string>

#if(GLM_COMPILER & GLM_COMPILER_CUDA)
#	error "GLM_GTX_string_cast is not supported on CUDA compiler"
#endif

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_string_cast extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL std::string to_string(genType const & x);

}

#include "string_cast.inl"

#endif
