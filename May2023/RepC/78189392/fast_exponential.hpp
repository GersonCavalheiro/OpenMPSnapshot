#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_fast_exponential extension included")
#endif
namespace glm
{
template <typename genType>
GLM_FUNC_DECL genType fastPow(genType x, genType y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastPow(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename genTypeT, typename genTypeU>
GLM_FUNC_DECL genTypeT fastPow(genTypeT x, genTypeU y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastPow(vecType<T, P> const & x);
template <typename T>
GLM_FUNC_DECL T fastExp(T x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastExp(vecType<T, P> const & x);
template <typename T>
GLM_FUNC_DECL T fastLog(T x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastLog(vecType<T, P> const & x);
template <typename T>
GLM_FUNC_DECL T fastExp2(T x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastExp2(vecType<T, P> const & x);
template <typename T>
GLM_FUNC_DECL T fastLog2(T x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastLog2(vecType<T, P> const & x);
}
#include "fast_exponential.inl"
