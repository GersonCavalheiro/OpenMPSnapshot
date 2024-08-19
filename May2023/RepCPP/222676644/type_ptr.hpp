
#ifndef GLM_GTC_type_ptr
#define GLM_GTC_type_ptr GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"
#include "../gtc/quaternion.hpp"
#include <cstring>

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_type_ptr extension included")
#endif

namespace glm
{ 

template<typename genType>
typename genType::value_type const * value_ptr(genType const & vec);

template<typename T>
detail::tvec2<T> make_vec2(T const * const ptr);

template<typename T>
detail::tvec3<T> make_vec3(T const * const ptr);

template<typename T>
detail::tvec4<T> make_vec4(T const * const ptr);

template<typename T>
detail::tmat2x2<T> make_mat2x2(T const * const ptr);

template<typename T>
detail::tmat2x3<T> make_mat2x3(T const * const ptr);

template<typename T>
detail::tmat2x4<T> make_mat2x4(T const * const ptr);

template<typename T>
detail::tmat3x2<T> make_mat3x2(T const * const ptr);

template<typename T>
detail::tmat3x3<T> make_mat3x3(T const * const ptr);

template<typename T>
detail::tmat3x4<T> make_mat3x4(T const * const ptr);

template<typename T>
detail::tmat4x2<T> make_mat4x2(
T const * const ptr);

template<typename T>
detail::tmat4x3<T> make_mat4x3(T const * const ptr);

template<typename T>
detail::tmat4x4<T> make_mat4x4(T const * const ptr);

template<typename T>
detail::tmat2x2<T> make_mat2(T const * const ptr);

template<typename T>
detail::tmat3x3<T> make_mat3(T const * const ptr);

template<typename T>
detail::tmat4x4<T> make_mat4(T const * const ptr);

template<typename T>
detail::tquat<T> make_quat(T const * const ptr);

}

#include "type_ptr.inl"

#endif

