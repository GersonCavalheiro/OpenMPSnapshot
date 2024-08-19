
#pragma once

#include "../geometric.hpp"
#include "../gtx/quaternion.hpp"
#include "../gtx/component_wise.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_norm is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_norm extension included")
#	endif
#endif

namespace glm
{

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL T length2(vec<L, T, Q> const& x);

template<length_t L, typename T, qualifier Q>
GLM_FUNC_DECL T distance2(vec<L, T, Q> const& p0, vec<L, T, Q> const& p1);

template<typename T, qualifier Q>
GLM_FUNC_DECL T l1Norm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

template<typename T, qualifier Q>
GLM_FUNC_DECL T l1Norm(vec<3, T, Q> const& v);

template<typename T, qualifier Q>
GLM_FUNC_DECL T l2Norm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

template<typename T, qualifier Q>
GLM_FUNC_DECL T l2Norm(vec<3, T, Q> const& x);

template<typename T, qualifier Q>
GLM_FUNC_DECL T lxNorm(vec<3, T, Q> const& x, vec<3, T, Q> const& y, unsigned int Depth);

template<typename T, qualifier Q>
GLM_FUNC_DECL T lxNorm(vec<3, T, Q> const& x, unsigned int Depth);

template<typename T, qualifier Q>
GLM_FUNC_DECL T lMaxNorm(vec<3, T, Q> const& x, vec<3, T, Q> const& y);

template<typename T, qualifier Q>
GLM_FUNC_DECL T lMaxNorm(vec<3, T, Q> const& x);

}

#include "norm.inl"
