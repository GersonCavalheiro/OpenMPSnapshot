
#ifndef GLM_GTX_mixed_product
#define GLM_GTX_mixed_product GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_mixed_product extension included")
#endif

namespace glm
{

template <typename valType> 
valType mixedProduct(
detail::tvec3<valType> const & v1, 
detail::tvec3<valType> const & v2, 
detail::tvec3<valType> const & v3);

}

#include "mixed_product.inl"

#endif
