
#ifndef GLM_GTX_normalize_dot
#define GLM_GTX_normalize_dot

#include "../glm.hpp"
#include "../gtx/fast_square_root.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_normalize_dot extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type normalizeDot(
genType const & x, 
genType const & y);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type fastNormalizeDot(
genType const & x, 
genType const & y);

}

#include "normalize_dot.inl"

#endif
