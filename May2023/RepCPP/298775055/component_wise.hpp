
#pragma once

#include "../detail/setup.hpp"
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_component_wise is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_component_wise extension included")
#	endif
#endif

namespace glm
{

template<typename floatType, length_t L, typename T, qualifier Q>
GLM_FUNC_DECL vec<L, floatType, Q> compNormalize(vec<L, T, Q> const& v);

template<length_t L, typename T, typename floatType, qualifier Q>
GLM_FUNC_DECL vec<L, T, Q> compScale(vec<L, floatType, Q> const& v);

template<typename genType>
GLM_FUNC_DECL typename genType::value_type compAdd(genType const& v);

template<typename genType>
GLM_FUNC_DECL typename genType::value_type compMul(genType const& v);

template<typename genType>
GLM_FUNC_DECL typename genType::value_type compMin(genType const& v);

template<typename genType>
GLM_FUNC_DECL typename genType::value_type compMax(genType const& v);

}

#include "component_wise.inl"
