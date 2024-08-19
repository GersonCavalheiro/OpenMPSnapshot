
#ifndef GLM_GTX_matrix_major_storage
#define GLM_GTX_matrix_major_storage GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_matrix_major_storage extension included")
#endif

namespace glm
{

template <typename T> 
detail::tmat2x2<T> rowMajor2(
detail::tvec2<T> const & v1, 
detail::tvec2<T> const & v2);

template <typename T> 
detail::tmat2x2<T> rowMajor2(
detail::tmat2x2<T> const & m);

template <typename T> 
detail::tmat3x3<T> rowMajor3(
detail::tvec3<T> const & v1, 
detail::tvec3<T> const & v2, 
detail::tvec3<T> const & v3);

template <typename T> 
detail::tmat3x3<T> rowMajor3(
detail::tmat3x3<T> const & m);

template <typename T> 
detail::tmat4x4<T> rowMajor4(
detail::tvec4<T> const & v1, 
detail::tvec4<T> const & v2,
detail::tvec4<T> const & v3, 
detail::tvec4<T> const & v4);

template <typename T> 
detail::tmat4x4<T> rowMajor4(
detail::tmat4x4<T> const & m);

template <typename T> 
detail::tmat2x2<T> colMajor2(
detail::tvec2<T> const & v1, 
detail::tvec2<T> const & v2);

template <typename T> 
detail::tmat2x2<T> colMajor2(
detail::tmat2x2<T> const & m);

template <typename T> 
detail::tmat3x3<T> colMajor3(
detail::tvec3<T> const & v1, 
detail::tvec3<T> const & v2, 
detail::tvec3<T> const & v3);

template <typename T> 
detail::tmat3x3<T> colMajor3(
detail::tmat3x3<T> const & m);

template <typename T> 
detail::tmat4x4<T> colMajor4(
detail::tvec4<T> const & v1, 
detail::tvec4<T> const & v2, 
detail::tvec4<T> const & v3, 
detail::tvec4<T> const & v4);

template <typename T> 
detail::tmat4x4<T> colMajor4(
detail::tmat4x4<T> const & m);

}

#include "matrix_major_storage.inl"

#endif
