#pragma once
#include "../common.hpp"
#include "../exponential.hpp"
#include "../geometric.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_fast_square_root extension included")
#endif
namespace glm
{
template <typename genType> 
GLM_FUNC_DECL genType fastSqrt(genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastSqrt(vecType<T, P> const & x);
template <typename genType> 
GLM_FUNC_DECL genType fastInverseSqrt(genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fastInverseSqrt(vecType<T, P> const & x);
template <typename genType>
GLM_FUNC_DECL genType fastLength(genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T fastLength(vecType<T, P> const & x);
template <typename genType>
GLM_FUNC_DECL genType fastDistance(genType x, genType y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T fastDistance(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename genType> 
GLM_FUNC_DECL genType fastNormalize(genType const & x);
}
#include "fast_square_root.inl"
