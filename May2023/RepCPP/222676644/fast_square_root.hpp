
#ifndef GLM_GTX_fast_square_root
#define GLM_GTX_fast_square_root GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_fast_square_root extension included")
#endif

namespace glm
{

template <typename genType> 
genType fastSqrt(genType const & x);

template <typename genType> 
genType fastInverseSqrt(genType const & x);

template <typename genType> 
typename genType::value_type fastLength(genType const & x);

template <typename genType> 
typename genType::value_type fastDistance(genType const & x, genType const & y);

template <typename genType> 
genType fastNormalize(genType const & x);

}

#include "fast_square_root.inl"

#endif
