#pragma once
#include "../detail/setup.hpp"
#if !GLM_HAS_TEMPLATE_ALIASES && !(GLM_COMPILER & GLM_COMPILER_GCC)
#	error "GLM_GTX_scalar_multiplication requires C++11 support or alias templates and if not support for GCC"
#endif
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../mat2x2.hpp"
#include <type_traits>
namespace glm
{
template <typename T, typename Vec>
using return_type_scalar_multiplication = typename std::enable_if<
!std::is_same<T, float>::value       
&& std::is_arithmetic<T>::value, Vec 
>::type;
#define GLM_IMPLEMENT_SCAL_MULT(Vec) \
template <typename T> \
return_type_scalar_multiplication<T, Vec> \
operator*(T const & s, Vec rh){ \
return rh *= static_cast<float>(s); \
} \
\
template <typename T> \
return_type_scalar_multiplication<T, Vec> \
operator*(Vec lh, T const & s){ \
return lh *= static_cast<float>(s); \
} \
\
template <typename T> \
return_type_scalar_multiplication<T, Vec> \
operator/(Vec lh, T const & s){ \
return lh *= 1.0f / s; \
}
GLM_IMPLEMENT_SCAL_MULT(vec2)
GLM_IMPLEMENT_SCAL_MULT(vec3)
GLM_IMPLEMENT_SCAL_MULT(vec4)
GLM_IMPLEMENT_SCAL_MULT(mat2)
GLM_IMPLEMENT_SCAL_MULT(mat2x3)
GLM_IMPLEMENT_SCAL_MULT(mat2x4)
GLM_IMPLEMENT_SCAL_MULT(mat3x2)
GLM_IMPLEMENT_SCAL_MULT(mat3)
GLM_IMPLEMENT_SCAL_MULT(mat3x4)
GLM_IMPLEMENT_SCAL_MULT(mat4x2)
GLM_IMPLEMENT_SCAL_MULT(mat4x3)
GLM_IMPLEMENT_SCAL_MULT(mat4)
#undef GLM_IMPLEMENT_SCAL_MULT
} 
