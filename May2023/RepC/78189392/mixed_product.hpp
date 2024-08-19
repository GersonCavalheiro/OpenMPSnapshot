#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_mixed_product extension included")
#endif
namespace glm
{
template <typename T, precision P> 
GLM_FUNC_DECL T mixedProduct(
tvec3<T, P> const & v1, 
tvec3<T, P> const & v2, 
tvec3<T, P> const & v3);
}
#include "mixed_product.inl"
