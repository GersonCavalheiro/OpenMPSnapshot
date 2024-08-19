
#pragma once

#include "../ext/scalar_ulp.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_ulp extension included")
#endif

namespace glm
{
template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextFloat(vec<L, T, Q> const& x);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextFloat(vec<L, T, Q> const& x, int ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextFloat(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevFloat(vec<L, T, Q> const& x);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevFloat(vec<L, T, Q> const& x, int ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevFloat(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, int, Q> floatDistance(vec<L, float, Q> const& x, vec<L, float, Q> const& y);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, int64, Q> floatDistance(vec<L, double, Q> const& x, vec<L, double, Q> const& y);

}

#include "vector_ulp.inl"
