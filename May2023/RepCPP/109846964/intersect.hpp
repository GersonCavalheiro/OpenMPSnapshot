
#ifndef GLM_GTX_intersect
#define GLM_GTX_intersect

#include "../glm.hpp"
#include "../gtx/closest_point.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_closest_point extension included")
#endif

namespace glm
{

template <typename genType>
GLM_FUNC_DECL bool intersectRayPlane(
genType const & orig, genType const & dir,
genType const & planeOrig, genType const & planeNormal,
typename genType::value_type & intersectionDistance);

template <typename genType>
GLM_FUNC_DECL bool intersectRayTriangle(
genType const & orig, genType const & dir,
genType const & vert0, genType const & vert1, genType const & vert2,
genType & baryPosition);

template <typename genType>
GLM_FUNC_DECL bool intersectLineTriangle(
genType const & orig, genType const & dir,
genType const & vert0, genType const & vert1, genType const & vert2,
genType & position);

template <typename genType>
GLM_FUNC_DECL bool intersectRaySphere(
genType const & rayStarting, genType const & rayNormalizedDirection,
genType const & sphereCenter, typename genType::value_type const sphereRadiusSquered,
typename genType::value_type & intersectionDistance);

template <typename genType>
GLM_FUNC_DECL bool intersectRaySphere(
genType const & rayStarting, genType const & rayNormalizedDirection,
genType const & sphereCenter, const typename genType::value_type sphereRadius,
genType & intersectionPosition, genType & intersectionNormal);

template <typename genType>
GLM_FUNC_DECL bool intersectLineSphere(
genType const & point0, genType const & point1,
genType const & sphereCenter, typename genType::value_type sphereRadius,
genType & intersectionPosition1, genType & intersectionNormal1, 
genType & intersectionPosition2 = genType(), genType & intersectionNormal2 = genType());

}

#include "intersect.inl"

#endif
