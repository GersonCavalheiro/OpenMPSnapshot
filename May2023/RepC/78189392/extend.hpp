#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_extend extension included")
#endif
namespace glm
{
template <typename genType> 
GLM_FUNC_DECL genType extend(
genType const & Origin, 
genType const & Source, 
typename genType::value_type const Length);
}
#include "extend.inl"
