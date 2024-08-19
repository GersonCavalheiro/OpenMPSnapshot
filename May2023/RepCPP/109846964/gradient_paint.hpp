
#ifndef GLM_GTX_gradient_paint
#define GLM_GTX_gradient_paint

#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_gradient_paint extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL T radialGradient(
detail::tvec2<T, P> const & Center,
T const & Radius,
detail::tvec2<T, P> const & Focal,
detail::tvec2<T, P> const & Position);

template <typename T, precision P>
GLM_FUNC_DECL T linearGradient(
detail::tvec2<T, P> const & Point0,
detail::tvec2<T, P> const & Point1,
detail::tvec2<T, P> const & Position);

}

#include "gradient_paint.inl"

#endif
