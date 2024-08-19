#pragma once
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/_noise.hpp"
#include "../geometric.hpp"
#include "../common.hpp"
#include "../vector_relational.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_noise extension included")
#endif
namespace glm
{
template <typename T, precision P, template<typename, precision> class vecType>
GLM_FUNC_DECL T perlin(
vecType<T, P> const & p);
template <typename T, precision P, template<typename, precision> class vecType>
GLM_FUNC_DECL T perlin(
vecType<T, P> const & p,
vecType<T, P> const & rep);
template <typename T, precision P, template<typename, precision> class vecType>
GLM_FUNC_DECL T simplex(
vecType<T, P> const & p);
}
#include "noise.inl"
