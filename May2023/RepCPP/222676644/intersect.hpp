
#ifndef GLM_GTX_intersect
#define GLM_GTX_intersect GLM_VERSION

#include "../glm.hpp"
#include "../gtx/closest_point.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{

template <typename genType>
bool intersectRayTriangle(
genType const & orig, genType const & dir,
genType const & vert0, genType const & vert1, genType const & vert2,
genType & baryPosition);

template <typename genType>
bool intersectLineTriangle(
genType const & orig, genType const & dir,
genType const & vert0, genType const & vert1, genType const & vert2,
genType & position);

template <typename genType>
bool intersectRaySphere(
genType const & rayStarting, genType const & rayNormalizedDirection,
genType const & sphereCenter, const typename genType::value_type sphereRadiusSquered,
typename genType::value_type & intersectionDistance);

template <typename genType>
bool intersectRaySphere(
genType const & rayStarting, genType const & rayNormalizedDirection,
genType const & sphereCenter, const typename genType::value_type sphereRadius,
genType & intersectionPosition, genType & intersectionNormal);

template <typename genType>
bool intersectLineSphere(
genType const & point0, genType const & point1,
genType const & sphereCenter, typename genType::value_type sphereRadius,
genType & intersectionPosition1, genType & intersectionNormal1, 
genType & intersectionPosition2 = genType(), genType & intersectionNormal2 = genType());

}

#include "intersect.inl"

#endif
