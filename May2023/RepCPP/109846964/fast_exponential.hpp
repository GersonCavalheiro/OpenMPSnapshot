
#ifndef GLM_GTX_fast_exponential
#define GLM_GTX_fast_exponential

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_fast_exponential extension included")
#endif

namespace glm
{

template <typename genType> 
GLM_FUNC_DECL genType fastPow(
genType const & x, 
genType const & y);

template <typename genTypeT, typename genTypeU> 
GLM_FUNC_DECL genTypeT fastPow(
genTypeT const & x, 
genTypeU const & y);

template <typename T> 
GLM_FUNC_DECL T fastExp(const T& x);

template <typename T> 
GLM_FUNC_DECL T fastLog(const T& x);

template <typename T> 
GLM_FUNC_DECL T fastExp2(const T& x);

template <typename T> 
GLM_FUNC_DECL T fastLog2(const T& x);

template <typename T> 
GLM_FUNC_DECL T fastLn(const T& x);

}

#include "fast_exponential.inl"

#endif
