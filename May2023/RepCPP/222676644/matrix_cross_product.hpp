
#ifndef GLM_GTX_matrix_cross_product
#define GLM_GTX_matrix_cross_product GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_matrix_cross_product extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat3x3<T> matrixCross3(
detail::tvec3<T> const & x);

template <typename T> 
detail::tmat4x4<T> matrixCross4(
detail::tvec3<T> const & x);

}

#include "matrix_cross_product.inl"

#endif
