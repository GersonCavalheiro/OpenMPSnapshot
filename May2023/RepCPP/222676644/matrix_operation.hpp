
#ifndef GLM_GTX_matrix_operation
#define GLM_GTX_matrix_operation GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_matrix_operation extension included")
#endif

namespace glm
{

template <typename valType> 
detail::tmat2x2<valType> diagonal2x2(
detail::tvec2<valType> const & v);

template <typename valType> 
detail::tmat2x3<valType> diagonal2x3(
detail::tvec2<valType> const & v);

template <typename valType> 
detail::tmat2x4<valType> diagonal2x4(
detail::tvec2<valType> const & v);

template <typename valType> 
detail::tmat3x2<valType> diagonal3x2(
detail::tvec2<valType> const & v);

template <typename valType> 
detail::tmat3x3<valType> diagonal3x3(
detail::tvec3<valType> const & v);

template <typename valType> 
detail::tmat3x4<valType> diagonal3x4(
detail::tvec3<valType> const & v);

template <typename valType> 
detail::tmat4x2<valType> diagonal4x2(
detail::tvec2<valType> const & v);

template <typename valType> 
detail::tmat4x3<valType> diagonal4x3(
detail::tvec3<valType> const & v);

template <typename valType> 
detail::tmat4x4<valType> diagonal4x4(
detail::tvec4<valType> const & v);

}

#include "matrix_operation.inl"

#endif
