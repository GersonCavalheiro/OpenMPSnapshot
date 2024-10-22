#pragma once
#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_spline extension included")
#endif
namespace glm
{
template <typename genType> 
GLM_FUNC_DECL genType catmullRom(
genType const & v1, 
genType const & v2, 
genType const & v3, 
genType const & v4, 
typename genType::value_type const & s);
template <typename genType> 
GLM_FUNC_DECL genType hermite(
genType const & v1, 
genType const & t1, 
genType const & v2, 
genType const & t2, 
typename genType::value_type const & s);
template <typename genType> 
GLM_FUNC_DECL genType cubic(
genType const & v1, 
genType const & v2, 
genType const & v3, 
genType const & v4, 
typename genType::value_type const & s);
}
#include "spline.inl"
