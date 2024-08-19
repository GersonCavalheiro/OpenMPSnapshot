
#ifndef GLM_GTX_matrix_interpolation
#define GLM_GTX_matrix_interpolation

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_interpolation extension included")
#endif

namespace glm
{

template <typename T, precision P>
GLM_FUNC_DECL void axisAngle(
detail::tmat4x4<T, P> const & mat,
detail::tvec3<T, P> & axis,
T & angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> axisAngleMatrix(
detail::tvec3<T, P> const & axis,
T const angle);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> extractMatrixRotation(
detail::tmat4x4<T, P> const & mat);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat4x4<T, P> interpolate(
detail::tmat4x4<T, P> const & m1,
detail::tmat4x4<T, P> const & m2,
T const delta);

}

#include "matrix_interpolation.inl"

#endif
