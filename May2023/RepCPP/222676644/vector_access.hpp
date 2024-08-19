
#ifndef GLM_GTX_vector_access
#define GLM_GTX_vector_access GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_vector_access extension included")
#endif

namespace glm
{

template <typename valType> 
void set(
detail::tvec2<valType> & v, 
valType const & x, 
valType const & y);

template <typename valType> 
void set(
detail::tvec3<valType> & v, 
valType const & x, 
valType const & y, 
valType const & z);

template <typename valType> 
void set(
detail::tvec4<valType> & v, 
valType const & x, 
valType const & y, 
valType const & z, 
valType const & w);

}

#include "vector_access.inl"

#endif
