
#ifndef glm_gtx_color_space_YCoCg
#define glm_gtx_color_space_YCoCg GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_color_space_YCoCg extension included")
#endif

namespace glm
{

template <typename valType> 
detail::tvec3<valType> rgb2YCoCg(
detail::tvec3<valType> const & rgbColor);

template <typename valType> 
detail::tvec3<valType> YCoCg2rgb(
detail::tvec3<valType> const & YCoCgColor);

template <typename valType> 
detail::tvec3<valType> rgb2YCoCgR(
detail::tvec3<valType> const & rgbColor);

template <typename valType> 
detail::tvec3<valType> YCoCgR2rgb(
detail::tvec3<valType> const & YCoCgColor);

}

#include "color_space_YCoCg.inl"

#endif
