#pragma once
#include "../gtc/bitfield.hpp"
#if(defined(GLM_MESSAGES))
#pragma message("GLM: GLM_GTX_bit extension is deprecated, include GLM_GTC_bitfield and GLM_GTC_integer instead")
#endif
namespace glm
{
template <typename genIUType>
GLM_FUNC_DECL genIUType highestBitValue(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> highestBitValue(vecType<T, P> const & value);
template <typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoAbove(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoAbove(vecType<T, P> const & value);
template <typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoBelow(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoBelow(vecType<T, P> const & value);
template <typename genIUType>
GLM_DEPRECATED GLM_FUNC_DECL genIUType powerOfTwoNearest(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_DEPRECATED GLM_FUNC_DECL vecType<T, P> powerOfTwoNearest(vecType<T, P> const & value);
} 
#include "bit.inl"
