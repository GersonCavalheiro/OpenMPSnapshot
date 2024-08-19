
#pragma once

#include "../detail/setup.hpp"
#include "../detail/qualifier.hpp"
#include "../detail/_vectorize.hpp"
#include "../vector_relational.hpp"
#include "../common.hpp"
#include <limits>

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_integer extension included")
#endif

namespace glm
{

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, bool, Q> isPowerOfTwo(vec<L, T, Q> const& v);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextPowerOfTwo(vec<L, T, Q> const& v);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevPowerOfTwo(vec<L, T, Q> const& v);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, bool, Q> isMultiple(vec<L, T, Q> const& v, T Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, bool, Q> isMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextMultiple(vec<L, T, Q> const& v, T Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> nextMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevMultiple(vec<L, T, Q> const& v, T Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> prevMultiple(vec<L, T, Q> const& v, vec<L, T, Q> const& Multiple);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, int, Q> findNSB(vec<L, T, Q> const& Source, vec<L, int, Q> SignificantBitCount);

} 

#include "vector_integer.inl"
