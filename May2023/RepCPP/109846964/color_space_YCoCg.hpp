
#ifndef glm_gtx_color_space_YCoCg
#define glm_gtx_color_space_YCoCg

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_color_space_YCoCg extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rgb2YCoCg(
detail::tvec3<T, P> const & rgbColor);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> YCoCg2rgb(
detail::tvec3<T, P> const & YCoCgColor);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> rgb2YCoCgR(
detail::tvec3<T, P> const & rgbColor);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> YCoCgR2rgb(
detail::tvec3<T, P> const & YCoCgColor);

}

#include "color_space_YCoCg.inl"

#endif
