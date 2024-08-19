
#ifndef GLM_GTX_rotate_vector
#define GLM_GTX_rotate_vector GLM_VERSION

#include "../glm.hpp"
#include "../gtx/transform.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_rotate_vector extension included")
#endif

namespace glm
{

template <typename T> 
detail::tvec2<T> rotate(
detail::tvec2<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec3<T> rotate(
detail::tvec3<T> const & v, 
T const & angle, 
detail::tvec3<T> const & normal);

template <typename T> 
detail::tvec4<T> rotate(
detail::tvec4<T> const & v, 
T const & angle, 
detail::tvec3<T> const & normal);

template <typename T> 
detail::tvec3<T> rotateX(
detail::tvec3<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec3<T> rotateY(
detail::tvec3<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec3<T> rotateZ(
detail::tvec3<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec4<T> rotateX(
detail::tvec4<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec4<T> rotateY(
detail::tvec4<T> const & v, 
T const & angle);

template <typename T> 
detail::tvec4<T> rotateZ(
detail::tvec4<T> const & v, 
T const & angle);

template <typename T> 
detail::tmat4x4<T> orientation(
detail::tvec3<T> const & Normal, 
detail::tvec3<T> const & Up);

}

#include "rotate_vector.inl"

#endif
