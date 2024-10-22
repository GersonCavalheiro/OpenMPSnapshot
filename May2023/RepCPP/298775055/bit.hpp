
#pragma once

#include "../gtc/bitfield.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_bit is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_bit extension included")
#	endif
#endif

namespace glm
{

template<typename genIUType>
GLM_FUNC_DECL genIUType highestBitValue(genIUType Value);

template<typename genIUType>
GLM_FUNC_DECL genIUType lowestBitValue(genIUType Value);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> highestBitValue(vec<L, T, Q> const& value);

template<typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoAbove(genIUType Value);

template<length_t L, typename T, qualifier Q>
GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoAbove(vec<L, T, Q> const& value);

template<typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoBelow(genIUType Value);

template<length_t L, typename T, qualifier Q>
GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoBelow(vec<L, T, Q> const& value);

template<typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoNearest(genIUType Value);

template<length_t L, typename T, qualifier Q>
GLM_DEPRECATED GLM_FUNC_DECL vec<L, T, Q> powerOfTwoNearest(vec<L, T, Q> const& value);

} 


#include "bit.inl"

