
#ifndef GLM_GTX_component_wise
#define GLM_GTX_component_wise

#include "../detail/setup.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_component_wise extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type compAdd(
genType const & v);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type compMul(
genType const & v);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type compMin(
genType const & v);

template <typename genType> 
GLM_FUNC_DECL typename genType::value_type compMax(
genType const & v);

}

#include "component_wise.inl"

#endif
