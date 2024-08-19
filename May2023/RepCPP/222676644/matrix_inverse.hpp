
#ifndef GLM_GTC_matrix_inverse
#define GLM_GTC_matrix_inverse GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{

template <typename genType> 
genType affineInverse(genType const & m);

template <typename genType> 
GLM_FUNC_QUALIFIER typename genType::value_type inverseTranspose(
genType const & m);

}

#include "matrix_inverse.inl"

#endif
