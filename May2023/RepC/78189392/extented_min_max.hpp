#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_extented_min_max extension included")
#endif
namespace glm
{
template <typename T>
GLM_FUNC_DECL T min(
T const & x, 
T const & y, 
T const & z);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const & x, 
typename C<T>::T const & y, 
typename C<T>::T const & z);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const & x, 
C<T> const & y, 
C<T> const & z);
template <typename T>
GLM_FUNC_DECL T min(
T const & x, 
T const & y, 
T const & z, 
T const & w);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const & x, 
typename C<T>::T const & y, 
typename C<T>::T const & z, 
typename C<T>::T const & w);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const & x, 
C<T> const & y, 
C<T> const & z,
C<T> const & w);
template <typename T>
GLM_FUNC_DECL T max(
T const & x, 
T const & y, 
T const & z);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const & x, 
typename C<T>::T const & y, 
typename C<T>::T const & z);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const & x, 
C<T> const & y, 
C<T> const & z);
template <typename T>
GLM_FUNC_DECL T max(
T const & x, 
T const & y, 
T const & z, 
T const & w);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const & x, 
typename C<T>::T const & y, 
typename C<T>::T const & z, 
typename C<T>::T const & w);
template <typename T, template <typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const & x, 
C<T> const & y, 
C<T> const & z, 
C<T> const & w);
}
#include "extented_min_max.inl"
