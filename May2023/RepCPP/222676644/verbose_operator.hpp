
#ifndef GLM_GTX_verbose_operator
#define GLM_GTX_verbose_operator GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_verbose_operator extension included")
#endif

namespace glm
{

template <typename genTypeT, typename genTypeU> 
genTypeT add(genTypeT const & a, genTypeU const & b);

template <typename genTypeT, typename genTypeU> 
genTypeT sub(genTypeT const & a, genTypeU const & b);

template <typename genTypeT, typename genTypeU> 
genTypeT mul(genTypeT const & a, genTypeU const & b);

template <typename genTypeT, typename genTypeU> 
genTypeT div(genTypeT const & a, genTypeU const & b);

template <typename genTypeT, typename genTypeU, typename genTypeV> 
genTypeT mad(genTypeT const & a, genTypeU const & b, genTypeV const & c);

}

#include "verbose_operator.inl"

#endif
