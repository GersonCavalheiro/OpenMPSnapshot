
#ifndef GLM_GTC_matrix_access
#define GLM_GTC_matrix_access GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_matrix_access extension included")
#endif

namespace glm
{

template <typename genType> 
typename genType::row_type row(
genType const & m, 
int index);

template <typename genType> 
genType row(
genType const & m, 
int index, 
typename genType::row_type const & x);

template <typename genType> 
typename genType::col_type column(
genType const & m, 
int index);

template <typename genType> 
genType column(
genType const & m, 
int index, 
typename genType::col_type const & x);

}

#include "matrix_access.inl"

#endif
