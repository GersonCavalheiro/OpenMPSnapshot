
#ifndef GLM_GTX_fast_square_root
#define GLM_GTX_fast_square_root

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_fast_square_root extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL genType fastSqrt(genType const & x);

template <typename genType> 
GLM_FUNC_DECL genType fastInverseSqrt(genType const & x);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastInverseSqrt(vecType<T, P> const & x);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type fastLength(genType const & x);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type fastDistance(genType const & x, genType const & y);

template <typename genType> 
GLM_FUNC_DECL genType fastNormalize(genType const & x);

}

#include "fast_square_root.inl"

#endif
