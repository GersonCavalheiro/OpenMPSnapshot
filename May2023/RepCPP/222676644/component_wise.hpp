
#ifndef GLM_GTX_component_wise
#define GLM_GTX_component_wise GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_component_wise extension included")
#endif

namespace glm
{

template <typename genType> 
typename genType::value_type compAdd(
genType const & v);

template <typename genType> 
typename genType::value_type compMul(
genType const & v);

template <typename genType> 
typename genType::value_type compMin(
genType const & v);

template <typename genType> 
typename genType::value_type compMax(
genType const & v);

}

#include "component_wise.inl"

#endif
