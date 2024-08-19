
#ifndef GLM_GTC_matrix_inverse
#define GLM_GTC_matrix_inverse

#include "../detail/setup.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL genType affineInverse(genType const & m);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type inverseTranspose(
genType const & m);

}

#include "matrix_inverse.inl"

#endif
