
#ifndef GLM_GTX_fast_trigonometry
#define GLM_GTX_fast_trigonometry GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_fast_trigonometry extension included")
#endif

namespace glm
{

template <typename T> 
T fastSin(const T& angle);

template <typename T> 
T fastCos(const T& angle);

template <typename T> 
T fastTan(const T& angle);

template <typename T> 
T fastAsin(const T& angle);

template <typename T> 
T fastAcos(const T& angle);

template <typename T> 
T fastAtan(const T& y, const T& x);

template <typename T> 
T fastAtan(const T& angle);

}

#include "fast_trigonometry.inl"

#endif
