
#ifndef GLM_GTX_fast_trigonometry
#define GLM_GTX_fast_trigonometry

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_fast_trigonometry extension included")
#endif

namespace glm
{

template <typename T> 
GLM_FUNC_DECL T fastSin(const T& angle);

template <typename T> 
GLM_FUNC_DECL T fastCos(const T& angle);

template <typename T> 
GLM_FUNC_DECL T fastTan(const T& angle);

template <typename T> 
GLM_FUNC_DECL T fastAsin(const T& angle);

template <typename T> 
GLM_FUNC_DECL T fastAcos(const T& angle);

template <typename T> 
GLM_FUNC_DECL T fastAtan(const T& y, const T& x);

template <typename T> 
GLM_FUNC_DECL T fastAtan(const T& angle);

}

#include "fast_trigonometry.inl"

#endif
