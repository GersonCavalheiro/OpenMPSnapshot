#pragma once
#include "../glm.hpp"
#include "../gtc/type_precision.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/dual_quaternion.hpp"
#include <string>
#if(GLM_COMPILER & GLM_COMPILER_CUDA)
#	error "GLM_GTX_string_cast is not supported on CUDA compiler"
#endif
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_string_cast extension included")
#endif
namespace glm
{
template <template <typename, precision> class matType, typename T, precision P>
GLM_FUNC_DECL std::string to_string(matType<T, P> const & x);
}
#include "string_cast.inl"
