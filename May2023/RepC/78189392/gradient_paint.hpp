#pragma once
#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_gradient_paint extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL T radialGradient(
tvec2<T, P> const & Center,
T const & Radius,
tvec2<T, P> const & Focal,
tvec2<T, P> const & Position);
template <typename T, precision P>
GLM_FUNC_DECL T linearGradient(
tvec2<T, P> const & Point0,
tvec2<T, P> const & Point1,
tvec2<T, P> const & Position);
}
#include "gradient_paint.inl"
