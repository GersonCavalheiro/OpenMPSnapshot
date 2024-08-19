#pragma once
#include "../gtc/constants.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_fast_trigonometry extension included")
#endif
namespace glm
{
template <typename T> 
GLM_FUNC_DECL T wrapAngle(T angle);
template <typename T>
GLM_FUNC_DECL T fastSin(T angle);
template <typename T> 
GLM_FUNC_DECL T fastCos(T angle);
template <typename T> 
GLM_FUNC_DECL T fastTan(T angle);
template <typename T> 
GLM_FUNC_DECL T fastAsin(T angle);
template <typename T> 
GLM_FUNC_DECL T fastAcos(T angle);
template <typename T> 
GLM_FUNC_DECL T fastAtan(T y, T x);
template <typename T> 
GLM_FUNC_DECL T fastAtan(T angle);
}
#include "fast_trigonometry.inl"
