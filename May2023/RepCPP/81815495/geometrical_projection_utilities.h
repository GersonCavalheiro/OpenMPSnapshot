
#pragma once



#include "geometries/geometry.h"
#include "includes/checks.h"
#include "includes/node.h"
#include "includes/variables.h"

namespace Kratos
{


class KRATOS_API(KRATOS_CORE) GeometricalProjectionUtilities
{
public:


enum class DistanceComputed
{
NO_RADIUS,
PROJECTION_ERROR,
RADIUS_PROJECTED,
RADIUS_NOT_PROJECTED_OUTSIDE,
RADIUS_NOT_PROJECTED_INSIDE
};

KRATOS_CLASS_POINTER_DEFINITION( GeometricalProjectionUtilities );

typedef Node                                              NodeType;
typedef Point                                               PointType;

typedef std::size_t                                         IndexType;

typedef std::size_t                                          SizeType;


GeometricalProjectionUtilities() = delete;



template<class TGeometryType>
static inline double FastProjectDirection(
const TGeometryType& rGeom,
const PointType& rPointToProject,
PointType& rPointProjected,
const array_1d<double,3>& rNormal,
const array_1d<double,3>& rVector,
const SizeType EchoLevel = 0
)
{
const double zero_tolerance = std::numeric_limits<double>::epsilon();

double distance = 0.0;

const array_1d<double,3> vector_points = rGeom[0].Coordinates() - rPointToProject.Coordinates();

if( norm_2( rVector ) < zero_tolerance && norm_2( rNormal ) > zero_tolerance ) {
distance = inner_prod(vector_points, rNormal)/norm_2(rNormal);
noalias(rPointProjected.Coordinates()) = rPointToProject.Coordinates() + rVector * distance;
KRATOS_WARNING_IF("GeometricalProjectionUtilities", EchoLevel > 0) << "WARNING:: Zero projection vector. Projection using the condition vector instead." << std::endl;
} else if (std::abs(inner_prod(rVector, rNormal) ) > zero_tolerance) {
distance = inner_prod(vector_points, rNormal)/inner_prod(rVector, rNormal);
noalias(rPointProjected.Coordinates()) = rPointToProject.Coordinates() + rVector * distance;
} else {
noalias(rPointProjected.Coordinates()) = rPointToProject.Coordinates();
KRATOS_WARNING_IF("GeometricalProjectionUtilities", EchoLevel > 0) << "WARNING:: The line and the plane are coplanar. Something wrong happened " << std::endl;
}

return distance;
}


template<class TPointClass1, class TPointClass2 = TPointClass1, class TPointClass3 = PointType>
static inline TPointClass3 FastProject(
const TPointClass1& rPointOrigin,
const TPointClass2& rPointToProject,
const array_1d<double,3>& rNormal,
double& rDistance
)
{
const array_1d<double,3> vector_points = rPointToProject - rPointOrigin;

rDistance = inner_prod(vector_points, rNormal);

TPointClass3 point_projected;

noalias(point_projected) = rPointToProject - rNormal * rDistance;

return point_projected;
}


template<class TGeometryType>
static inline double FastProjectOnGeometry(const TGeometryType& rGeom,
const Point& rPointToProject,
PointType& rPointProjected,
const SizeType EchoLevel = 0)
{
array_1d<double,3> local_coords_center;
rGeom.PointLocalCoordinates(local_coords_center, rGeom.Center());
const array_1d<double,3> normal = rGeom.UnitNormal(local_coords_center);

return FastProjectDirection(
rGeom,
rPointToProject,
rPointProjected,
normal,
normal,
EchoLevel);
}


template<class TGeometryType>
static inline double FastProjectOnLine(const TGeometryType& rGeometry,
const PointType& rPointToProject,
PointType& rPointProjected)
{
const array_1d<double, 3>& r_p_a = rGeometry[0].Coordinates();
const array_1d<double, 3>& r_p_b = rGeometry[1].Coordinates();
const array_1d<double, 3> ab = r_p_b - r_p_a;

const array_1d<double, 3>& p_c = rPointToProject.Coordinates();

const double factor = (inner_prod(r_p_b, p_c) - inner_prod(r_p_a, p_c) - inner_prod(r_p_b, r_p_a) + inner_prod(r_p_a, r_p_a)) / inner_prod(ab, ab);

rPointProjected.Coordinates() = r_p_a + factor * ab;

return norm_2(rPointProjected.Coordinates()-p_c);
}



template<class TGeometryType>
static inline double FastMinimalDistanceOnLine(
const TGeometryType& rGeometry,
const PointType& rPoint,
const double Tolerance = 1.0e-9
)
{
PointType projected_point;
const double projected_distance = FastProjectOnLine(rGeometry, rPoint, projected_point);
typename TGeometryType::CoordinatesArrayType projected_local;
if (rGeometry.IsInside(projected_point.Coordinates(), projected_local, Tolerance)) {
return projected_distance;
} else {
const double distance_a = rPoint.Distance(rGeometry[0]);
const double distance_b = rPoint.Distance(rGeometry[1]);
return std::min(distance_a, distance_b);
}
}


static DistanceComputed FastMinimalDistanceOnLineWithRadius(
double& rDistance,
const Geometry<Node>& rSegment,
const Point& rPoint,
const double Radius,
const double Tolerance = 1.0e-9
);


template<class TGeometryType, class TPointClass1, class TPointClass2 = TPointClass1>
static inline double FastProjectOnLine2D(
const TGeometryType& rGeometry,
const TPointClass1& rPointToProject,
TPointClass2& rPointProjected
)
{
const array_1d<double, 3>& r_p_a = rGeometry[0];
const array_1d<double, 3>& r_p_b = rGeometry[1];
const array_1d<double, 3>& r_p_c = rPointToProject;

array_1d<double,3> normal;
normal[0] = r_p_b[1] - r_p_a[1];
normal[1] = r_p_a[0] - r_p_b[0];
normal[2] = 0.0;

const double norm_normal = norm_2(normal);
KRATOS_ERROR_IF(norm_normal <= std::numeric_limits<double>::epsilon()) << "Zero norm normal: X: " << normal[0] << "\tY: " << normal[1] << std::endl;
normal /= norm_normal;

const array_1d<double,3> vector_points = r_p_a - r_p_c;
const double distance = inner_prod(vector_points, normal);
noalias(rPointProjected) = r_p_c + normal * distance;

return distance;
}


template<class TGeometryType>
static inline bool ProjectIterativeLine2D(
TGeometryType& rGeomOrigin,
const array_1d<double,3>& rPointDestiny,
array_1d<double,3>& rResultingPoint,
const array_1d<double, 3>& rNormal,
const double Tolerance = 1.0e-8,
double DeltaXi = 0.5
)
{

double old_delta_xi = 0.0;

array_1d<double, 3> current_global_coords;

KRATOS_DEBUG_CHECK_VARIABLE_IN_NODAL_DATA(NORMAL, rGeomOrigin[0])
KRATOS_DEBUG_CHECK_VARIABLE_IN_NODAL_DATA(NORMAL, rGeomOrigin[1])

array_1d<array_1d<double, 3>, 2> normals;
normals[0] = rGeomOrigin[0].FastGetSolutionStepValue(NORMAL);
normals[1] = rGeomOrigin[1].FastGetSolutionStepValue(NORMAL);

BoundedMatrix<double,2,2> X;
BoundedMatrix<double,2,1> DN;
for(unsigned int i=0; i<2;++i) {
X(0,i) = rGeomOrigin[i].X();
X(1,i) = rGeomOrigin[i].Y();
}

Matrix J = ZeroMatrix( 1, 1 );


const unsigned int max_iter = 20;

for ( unsigned int k = 0; k < max_iter; ++k ) {
array_1d<double, 2> N_origin;
N_origin[0] = 0.5 * ( 1.0 - rResultingPoint[0]);
N_origin[1] = 0.5 * ( 1.0 + rResultingPoint[0]);

array_1d<double,3> normal_xi = N_origin[0] * normals[0] + N_origin[1] * normals[1];
normal_xi = normal_xi/norm_2(normal_xi);

current_global_coords = ZeroVector(3);
for( IndexType i_node = 0; i_node < 2; ++i_node )
current_global_coords += N_origin[i_node] * rGeomOrigin[i_node].Coordinates();

const array_1d<double,3> VectorPoints = rGeomOrigin.Center() - rPointDestiny;
const double distance = inner_prod(VectorPoints, rNormal)/inner_prod(-normal_xi, rNormal);
const array_1d<double, 3> current_destiny_global_coords = rPointDestiny - normal_xi * distance;

Matrix ShapeFunctionsGradients;
ShapeFunctionsGradients = rGeomOrigin.ShapeFunctionsLocalGradients(ShapeFunctionsGradients, rResultingPoint );

noalias(DN) = prod(X,ShapeFunctionsGradients);

noalias(J) = prod(trans(DN),DN); 

const Vector RHS = prod(trans(DN),subrange(current_destiny_global_coords - current_global_coords,0,2));

old_delta_xi = DeltaXi;
DeltaXi = RHS[0]/J(0, 0);

rResultingPoint[0] += DeltaXi;

if (rResultingPoint[0] <= -1.0)
rResultingPoint[0] = -1.0;
else if (rResultingPoint[0] >= 1.0)
rResultingPoint[0] = 1.0;

if ( std::abs(DeltaXi - old_delta_xi) < Tolerance )
return true;
}

return false;
}

private:
};


} 
