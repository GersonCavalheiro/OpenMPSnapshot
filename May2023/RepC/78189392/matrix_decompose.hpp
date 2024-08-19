#pragma once
#include "../mat4x4.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtc/matrix_transform.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_matrix_decompose extension included")
#endif
namespace glm
{
template <typename T, precision P>
GLM_FUNC_DECL bool decompose(
tmat4x4<T, P> const & modelMatrix,
tvec3<T, P> & scale, tquat<T, P> & orientation, tvec3<T, P> & translation, tvec3<T, P> & skew, tvec4<T, P> & perspective);
}
#include "matrix_decompose.inl"
