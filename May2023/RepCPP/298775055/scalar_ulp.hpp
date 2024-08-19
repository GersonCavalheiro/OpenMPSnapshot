
#pragma once

#include "../ext/scalar_int_sized.hpp"
#include "../common.hpp"
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_scalar_ulp extension included")
#endif

namespace glm
{
template<typename genType>
GLM_FUNC_DECL genType nextFloat(genType x);

template<typename genType>
GLM_FUNC_DECL genType prevFloat(genType x);

template<typename genType>
GLM_FUNC_DECL genType nextFloat(genType x, int ULPs);

template<typename genType>
GLM_FUNC_DECL genType prevFloat(genType x, int ULPs);

GLM_FUNC_DECL int floatDistance(float x, float y);

GLM_FUNC_DECL int64 floatDistance(double x, double y);

}

#include "scalar_ulp.inl"
