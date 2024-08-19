
#pragma once

#include "../detail/setup.hpp"
#include "../detail/qualifier.hpp"
#include "../detail/_vectorize.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_ulp extension included")
#endif

namespace glm
{
template<typename genType>
GLM_FUNC_DECL genType next_float(genType x);

template<typename genType>
GLM_FUNC_DECL genType prev_float(genType x);

template<typename genType>
GLM_FUNC_DECL genType next_float(genType x, int ULPs);

template<typename genType>
GLM_FUNC_DECL genType prev_float(genType x, int ULPs);

GLM_FUNC_DECL int float_distance(float x, float y);

GLM_FUNC_DECL int64 float_distance(double x, double y);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> next_float(vec<L, T, Q> const& x);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> next_float(vec<L, T, Q> const& x, int ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> next_float(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prev_float(vec<L, T, Q> const& x);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prev_float(vec<L, T, Q> const& x, int ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prev_float(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, int, Q> float_distance(vec<L, float, Q> const& x, vec<L, float, Q> const& y);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, int64, Q> float_distance(vec<L, double, Q> const& x, vec<L, double, Q> const& y);

}

#include "ulp.inl"
