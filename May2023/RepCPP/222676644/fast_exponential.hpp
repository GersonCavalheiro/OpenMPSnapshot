
#ifndef GLM_GTX_fast_exponential
#define GLM_GTX_fast_exponential GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_fast_exponential extension included")
#endif

namespace glm
{

template <typename genType> 
genType fastPow(
genType const & x, 
genType const & y);

template <typename genTypeT, typename genTypeU> 
genTypeT fastPow(
genTypeT const & x, 
genTypeU const & y);

template <typename T> 
T fastExp(const T& x);

template <typename T> 
T fastLog(const T& x);

template <typename T> 
T fastExp2(const T& x);

template <typename T> 
T fastLog2(const T& x);

template <typename T> 
T fastLn(const T& x);

}

#include "fast_exponential.inl"

#endif
