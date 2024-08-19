
#ifndef GLM_GTX_mixed_product
#define GLM_GTX_mixed_product

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_mixed_product extension included")
#endif

namespace glm
{

template <typename T, precision P> 
GLM_FUNC_DECL T mixedProduct(
detail::tvec3<T, P> const & v1, 
detail::tvec3<T, P> const & v2, 
detail::tvec3<T, P> const & v3);

}

#include "mixed_product.inl"

#endif
