
#ifndef GLM_GTX_vector_query
#define GLM_GTX_vector_query GLM_VERSION

#include "../glm.hpp"
#include <cfloat>
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_vector_query extension included")
#endif

namespace glm
{

template <typename genType> 
bool areCollinear(
genType const & v0, 
genType const & v1, 
typename genType::value_type const & epsilon);

template <typename genType> 
bool areOrthogonal(
genType const & v0, 
genType const & v1, 
typename genType::value_type const & epsilon);

template <typename genType, template <typename> class vecType> 
bool isNormalized(
vecType<genType> const & v, 
genType const & epsilon);

template <typename valType> 
bool isNull(
detail::tvec2<valType> const & v, 
valType const & epsilon);

template <typename valType> 
bool isNull(
detail::tvec3<valType> const & v, 
valType const & epsilon);

template <typename valType> 
bool isNull(
detail::tvec4<valType> const & v, 
valType const & epsilon);

template <typename genType>
bool areOrthonormal(
genType const & v0, 
genType const & v1, 
typename genType::value_type const & epsilon);

}

#include "vector_query.inl"

#endif
