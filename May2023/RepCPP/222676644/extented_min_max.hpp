
#ifndef GLM_GTX_extented_min_max
#define GLM_GTX_extented_min_max GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_extented_min_max extension included")
#endif

namespace glm
{

template <typename T>
T min(
T const & x, 
T const & y, 
T const & z);

template 
<
typename T, 
template <typename> class C
>
C<T> min(
C<T> const & x, 
typename C<T>::value_type const & y, 
typename C<T>::value_type const & z);

template 
<
typename T, 
template <typename> class C
>
C<T> min(
C<T> const & x, 
C<T> const & y, 
C<T> const & z);

template <typename T>
T min(
T const & x, 
T const & y, 
T const & z, 
T const & w);

template 
<
typename T, 
template <typename> class C
>
C<T> min(
C<T> const & x, 
typename C<T>::value_type const & y, 
typename C<T>::value_type const & z, 
typename C<T>::value_type const & w);

template 
<
typename T, 
template <typename> class C
>
C<T> min(
C<T> const & x, 
C<T> const & y, 
C<T> const & z,
C<T> const & w);

template <typename T>
T max(
T const & x, 
T const & y, 
T const & z);

template 
<
typename T, 
template <typename> class C
>
C<T> max(
C<T> const & x, 
typename C<T>::value_type const & y, 
typename C<T>::value_type const & z);

template 
<
typename T, 
template <typename> class C
>
C<T> max(
C<T> const & x, 
C<T> const & y, 
C<T> const & z);

template <typename T>
T max(
T const & x, 
T const & y, 
T const & z, 
T const & w);

template 
<
typename T, 
template <typename> class C
>
C<T> max(
C<T> const & x, 
typename C<T>::value_type const & y, 
typename C<T>::value_type const & z, 
typename C<T>::value_type const & w);

template 
<
typename T, 
template <typename> class C
>
C<T> max(
C<T> const & x, 
C<T> const & y, 
C<T> const & z, 
C<T> const & w);

}

#include "extented_min_max.inl"

#endif
