
#pragma once

#include "../glm.hpp"
#include "../ext/vector_common.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_extented_min_max is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_extented_min_max extension included")
#	endif
#endif

namespace glm
{

template<typename T>
GLM_FUNC_DECL T min(
T const& x,
T const& y,
T const& z);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const& x,
typename C<T>::T const& y,
typename C<T>::T const& z);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const& x,
C<T> const& y,
C<T> const& z);

template<typename T>
GLM_FUNC_DECL T min(
T const& x,
T const& y,
T const& z,
T const& w);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const& x,
typename C<T>::T const& y,
typename C<T>::T const& z,
typename C<T>::T const& w);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> min(
C<T> const& x,
C<T> const& y,
C<T> const& z,
C<T> const& w);

template<typename T>
GLM_FUNC_DECL T max(
T const& x,
T const& y,
T const& z);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const& x,
typename C<T>::T const& y,
typename C<T>::T const& z);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const& x,
C<T> const& y,
C<T> const& z);

template<typename T>
GLM_FUNC_DECL T max(
T const& x,
T const& y,
T const& z,
T const& w);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const& x,
typename C<T>::T const& y,
typename C<T>::T const& z,
typename C<T>::T const& w);

template<typename T, template<typename> class C>
GLM_FUNC_DECL C<T> max(
C<T> const& x,
C<T> const& y,
C<T> const& z,
C<T> const& w);

}

#include "extended_min_max.inl"
