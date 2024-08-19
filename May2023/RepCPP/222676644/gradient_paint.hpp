
#ifndef GLM_GTX_gradient_paint
#define GLM_GTX_gradient_paint GLM_VERSION

#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_gradient_paint extension included")
#endif

namespace glm
{

template <typename valType>
valType radialGradient(
detail::tvec2<valType> const & Center,
valType const & Radius,
detail::tvec2<valType> const & Focal,
detail::tvec2<valType> const & Position);

template <typename valType>
valType linearGradient(
detail::tvec2<valType> const & Point0,
detail::tvec2<valType> const & Point1,
detail::tvec2<valType> const & Position);

}

#include "gradient_paint.inl"

#endif
