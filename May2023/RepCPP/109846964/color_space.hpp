
#ifndef GLM_GTX_color_space
#define GLM_GTX_color_space

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_color_space extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rgbColor(
detail::tvec3<T, P> const & hsvValue);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> hsvColor(
detail::tvec3<T, P> const & rgbValue);

template <typename T>
GLM_FUNC_DECL detail::tmat4x4<T, defaultp> saturation(
T const s);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> saturation(
T const s,
detail::tvec3<T, P> const & color);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> saturation(
T const s,
detail::tvec4<T, P> const & color);

template <typename T, precision P>
GLM_FUNC_DECL T luminosity(
detail::tvec3<T, P> const & color);

}

#include "color_space.inl"

#endif
