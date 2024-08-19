
#ifndef GLM_GTX_matrix_interpolation
#define GLM_GTX_matrix_interpolation GLM_VERSION


#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_matrix_interpolation extension included")
#endif

namespace glm
{

template <typename T>
void axisAngle(
detail::tmat4x4<T> const & mat,
detail::tvec3<T> & axis,
T & angle);

template <typename T>
detail::tmat4x4<T> axisAngleMatrix(
detail::tvec3<T> const & axis,
T const angle);

template <typename T>
detail::tmat4x4<T> extractMatrixRotation(
detail::tmat4x4<T> const & mat);

template <typename T>
detail::tmat4x4<T> interpolate(
detail::tmat4x4<T> const & m1,
detail::tmat4x4<T> const & m2,
T const delta);

}

#include "matrix_interpolation.inl"

#endif
