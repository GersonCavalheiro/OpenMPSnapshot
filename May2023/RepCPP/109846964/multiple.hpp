
#ifndef GLM_GTX_multiple
#define GLM_GTX_multiple

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_multiple extension included")
#endif

namespace glm
{

template <typename genType>
GLM_FUNC_DECL genType higherMultiple(
genType const & Source,
genType const & Multiple);

template <typename genType>
GLM_FUNC_DECL genType lowerMultiple(
genType const & Source,
genType const & Multiple);

}

#include "multiple.inl"

#endif
