
#ifndef GLM_GTX_matrix_query
#define GLM_GTX_matrix_query GLM_VERSION

#include "../glm.hpp"
#include "../gtx/vector_query.hpp"
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_matrix_query extension included")
#endif

namespace glm
{

template<typename T> 
bool isNull(
detail::tmat2x2<T> const & m, 
T const & epsilon);

template<typename T> 
bool isNull(
detail::tmat3x3<T> const & m, 
T const & epsilon);

template<typename T> 
bool isNull(
detail::tmat4x4<T> const & m, 
T const & epsilon);

template<typename genType> 
bool isIdentity(
genType const & m, 
typename genType::value_type const & epsilon);

template<typename valType>   
bool isNormalized(
detail::tmat2x2<valType> const & m, 
valType const & epsilon);

template<typename valType>   
bool isNormalized(
detail::tmat3x3<valType> const & m, 
valType const & epsilon);

template<typename valType>   
bool isNormalized(
detail::tmat4x4<valType> const & m, 
valType const & epsilon);

template<typename valType, template <typename> class matType> 
bool isOrthogonal(
matType<valType> const & m, 
valType const & epsilon);

}

#include "matrix_query.inl"

#endif
