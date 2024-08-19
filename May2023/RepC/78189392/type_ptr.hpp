#pragma once
#include "../gtc/quaternion.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"
#include <cstring>
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_type_ptr extension included")
#endif
namespace glm
{
template<typename genType>
GLM_FUNC_DECL typename genType::value_type const * value_ptr(genType const & vec);
template<typename T>
GLM_FUNC_DECL tvec2<T, defaultp> make_vec2(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tvec3<T, defaultp> make_vec3(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tvec4<T, defaultp> make_vec4(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat2x2<T, defaultp> make_mat2x2(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat2x3<T, defaultp> make_mat2x3(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat2x4<T, defaultp> make_mat2x4(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat3x2<T, defaultp> make_mat3x2(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat3x3<T, defaultp> make_mat3x3(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat3x4<T, defaultp> make_mat3x4(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat4x2<T, defaultp> make_mat4x2(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat4x3<T, defaultp> make_mat4x3(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> make_mat4x4(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat2x2<T, defaultp> make_mat2(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat3x3<T, defaultp> make_mat3(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> make_mat4(T const * const ptr);
template<typename T>
GLM_FUNC_DECL tquat<T, defaultp> make_quat(T const * const ptr);
}
#include "type_ptr.inl"
