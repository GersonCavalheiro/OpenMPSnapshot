
#pragma once

#include "../glm.hpp"
#include "../gtx/optimum_pow.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_spline is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_spline extension included")
#	endif
#endif

namespace glm
{

template<typename genType>
GLM_FUNC_DECL genType catmullRom(
genType const& v1,
genType const& v2,
genType const& v3,
genType const& v4,
typename genType::value_type const& s);

template<typename genType>
GLM_FUNC_DECL genType hermite(
genType const& v1,
genType const& t1,
genType const& v2,
genType const& t2,
typename genType::value_type const& s);

template<typename genType>
GLM_FUNC_DECL genType cubic(
genType const& v1,
genType const& v2,
genType const& v3,
genType const& v4,
typename genType::value_type const& s);

}

#include "spline.inl"
