
#ifndef GLM_GTX_spline
#define GLM_GTX_spline GLM_VERSION

#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_spline extension included")
#endif

namespace glm
{

template <typename genType> 
genType catmullRom(
genType const & v1, 
genType const & v2, 
genType const & v3, 
genType const & v4, 
typename genType::value_type const & s);

template <typename genType> 
genType hermite(
genType const & v1, 
genType const & t1, 
genType const & v2, 
genType const & t2, 
typename genType::value_type const & s);

template <typename genType> 
genType cubic(
genType const & v1, 
genType const & v2, 
genType const & v3, 
genType const & v4, 
typename genType::value_type const & s);

}

#include "spline.inl"

#endif

