#pragma once
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../gtc/vec1.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_common extension included")
#endif
namespace glm
{
template <typename genType> 
GLM_FUNC_DECL typename genType::bool_type isdenormal(genType const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fmod(vecType<T, P> const & v);
}
#include "common.inl"
