
#ifndef GLM_GTX_color_space
#define GLM_GTX_color_space GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_color_space extension included")
#endif

namespace glm
{

template <typename valType> 
detail::tvec3<valType> rgbColor(
detail::tvec3<valType> const & hsvValue);

template <typename valType> 
detail::tvec3<valType> hsvColor(
detail::tvec3<valType> const & rgbValue);

template <typename valType> 
detail::tmat4x4<valType> saturation(
valType const s);

template <typename valType> 
detail::tvec3<valType> saturation(
valType const s, 
detail::tvec3<valType> const & color);

template <typename valType> 
detail::tvec4<valType> saturation(
valType const s, 
detail::tvec4<valType> const & color);

template <typename valType> 
valType luminosity(
detail::tvec3<valType> const & color);

}

#include "color_space.inl"

#endif
