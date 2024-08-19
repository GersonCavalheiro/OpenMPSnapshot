
#pragma once

#include <iomanip>


#include "geometries/plane_3d.h"
#include "geometries/line_3d_2.h"
#include "integration/triangle_gauss_legendre_integration_points.h"
#include "integration/triangle_collocation_integration_points.h"
#include "utilities/geometry_utilities.h"
#include "utilities/geometrical_projection_utilities.h"
#include "utilities/intersection_utilities.h"

namespace Kratos
{






template<class TPointType> class Triangle3D3
: public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;

typedef Geometry<TPointType> GeometryType;


typedef Line3D2<TPointType> EdgeType;


typedef Triangle3D3<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Triangle3D3 );


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef typename BaseType::GeometriesArrayType GeometriesArrayType;


typedef TPointType PointType;


typedef typename BaseType::IndexType IndexType;


typedef typename BaseType::SizeType SizeType;


typedef  typename BaseType::PointsArrayType PointsArrayType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;


typedef typename BaseType::IntegrationPointType IntegrationPointType;


typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


typedef typename BaseType::IntegrationPointsContainerType IntegrationPointsContainerType;


typedef typename BaseType::ShapeFunctionsValuesContainerType ShapeFunctionsValuesContainerType;


typedef typename BaseType::ShapeFunctionsLocalGradientsContainerType ShapeFunctionsLocalGradientsContainerType;


typedef typename BaseType::JacobiansType JacobiansType;


typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef typename BaseType::ShapeFunctionsSecondDerivativesType
ShapeFunctionsSecondDerivativesType;


typedef typename BaseType::ShapeFunctionsThirdDerivativesType
ShapeFunctionsThirdDerivativesType;


typedef typename BaseType::NormalType NormalType;



Triangle3D3( typename PointType::Pointer pFirstPoint,
typename PointType::Pointer pSecondPoint,
typename PointType::Pointer pThirdPoint )
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( pFirstPoint );
this->Points().push_back( pSecondPoint );
this->Points().push_back( pThirdPoint );
}

explicit Triangle3D3( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF(this->PointsNumber() != 3) << "Invalid points number. Expected 3, given " << this->PointsNumber() << std::endl;
}

explicit Triangle3D3(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 3) << "Invalid points number. Expected 3, given " << this->PointsNumber() << std::endl;
}

explicit Triangle3D3(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 3) << "Invalid points number. Expected 3, given " << this->PointsNumber() << std::endl;
}


Triangle3D3( Triangle3D3 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> explicit Triangle3D3( Triangle3D3<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Triangle3D3() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Triangle;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Triangle3D3;
}



Triangle3D3& operator=( const Triangle3D3& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Triangle3D3& operator=( Triangle3D3<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );
return *this;
}



typename BaseType::Pointer Create(
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Triangle3D3( rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Triangle3D3( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Triangle3D3( rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Triangle3D3( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
rResult.resize( 3, 2 ,false);
noalias( rResult ) = ZeroMatrix( 3, 2 );
rResult( 0, 0 ) =  0.0;
rResult( 0, 1 ) =  0.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) =  0.0;
rResult( 2, 0 ) =  0.0;
rResult( 2, 1 ) =  1.0;
return rResult;
}


Vector& LumpingFactors(
Vector& rResult,
const typename BaseType::LumpingMethods LumpingMethod = BaseType::LumpingMethods::ROW_SUM
)  const override
{
rResult.resize( 3, false );
std::fill( rResult.begin(), rResult.end(), 1.00 / 3.00 );
return rResult;
}




double Length() const override
{
return std::sqrt(2.0 * Area());
}


double Area() const override
{
const double a = MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1));
const double b = MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2));
const double c = MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0));

const double s = (a+b+c) / 2.0;

return std::sqrt(s*(s-a)*(s-b)*(s-c));
}



double DomainSize() const override
{
return Area();
}



double MinEdgeLength() const override
{
const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

return CalculateMinEdgeLength(sa, sb, sc);
}


double MaxEdgeLength() const override
{
const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

return CalculateMaxEdgeLength(sa, sb, sc);
}


double AverageEdgeLength() const override
{
return CalculateAvgEdgeLength(
MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1)),
MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2)),
MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0))
);
}


double Circumradius() const override
{
return CalculateCircumradius(
MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1)),
MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2)),
MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0))
);
}


double Inradius() const override
{
return CalculateInradius(
MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1)),
MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2)),
MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0))
);
}


bool AllSameSide(array_1d<double, 3> const& Distances) const
{
constexpr double epsilon = std::numeric_limits<double>::epsilon();

double du0 = Distances[0];
double du1 = Distances[1];
double du2 = Distances[2];

if (std::abs(du0)<epsilon) du0 = 0.0;
if (std::abs(du1)<epsilon) du1 = 0.0;
if (std::abs(du2)<epsilon) du2 = 0.0;

const double du0du1 = du0*du1;
const double du0du2 = du0*du2;

if (du0du1>0.00 && du0du2>0.00)
return true;                   

return false;

}

int GetMajorAxis(array_1d<double, 3> const& V) const
{
int index = static_cast<int>(std::abs(V[0]) < std::abs(V[1]));
return (std::abs(V[index]) > std::abs(V[2])) ? index : 2;
}


bool HasIntersection(const GeometryType& rThisGeometry) const override
{
const auto geometry_type = rThisGeometry.GetGeometryType();

if (geometry_type == GeometryData::KratosGeometryType::Kratos_Line3D2) {
return LineTriangleOverlap(rThisGeometry[0], rThisGeometry[1]);
}
else if(geometry_type == GeometryData::KratosGeometryType::Kratos_Triangle3D3) {
return TriangleTriangleOverlap(rThisGeometry[0], rThisGeometry[1], rThisGeometry[2]);
}
else if(geometry_type == GeometryData::KratosGeometryType::Kratos_Quadrilateral3D4) {
if      ( TriangleTriangleOverlap(rThisGeometry[0], rThisGeometry[1], rThisGeometry[2]) ) return true;
else if ( TriangleTriangleOverlap(rThisGeometry[2], rThisGeometry[3], rThisGeometry[0]) ) return true;
else return false;
}
else {
KRATOS_ERROR << "Triangle3D3::HasIntersection : Geometry cannot be identified, please, check the intersecting geometry type." << std::endl;
}
}


bool HasIntersection( const Point& rLowPoint, const Point& rHighPoint) const override
{
Point box_center;
Point box_half_size;

box_center[0] = 0.5 * (rLowPoint[0] + rHighPoint[0]);
box_center[1] = 0.5 * (rLowPoint[1] + rHighPoint[1]);
box_center[2] = 0.5 * (rLowPoint[2] + rHighPoint[2]);

box_half_size[0] = 0.5 * std::abs(rHighPoint[0] - rLowPoint[0]);
box_half_size[1] = 0.5 * std::abs(rHighPoint[1] - rLowPoint[1]);
box_half_size[2] = 0.5 * std::abs(rHighPoint[2] - rLowPoint[2]);

return TriBoxOverlap(box_center, box_half_size);
}



double InradiusToCircumradiusQuality() const override
{
constexpr double normFactor = 1.0;

const double a = MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1));
const double b = MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2));
const double c = MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0));

return normFactor * CalculateInradius(a,b,c) / CalculateCircumradius(a,b,c);
};


double InradiusToLongestEdgeQuality() const override
{
constexpr double normFactor = 1.0; 

const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

return normFactor * CalculateInradius(std::sqrt(sa),std::sqrt(sb),std::sqrt(sc)) / CalculateMaxEdgeLength(sa,sb,sc);
}


double AreaToEdgeLengthRatio() const override
{
constexpr double normFactor = 1.0;

const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

return normFactor * Area() / (sa+sb+sc);
}


double ShortestAltitudeToEdgeLengthRatio() const override {
constexpr double normFactor = 1.0;

const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

double base = CalculateMaxEdgeLength(sa,sb,sc);

return normFactor * (Area() * 2 / base ) / std::sqrt(sa+sb+sc);
}


virtual double AreaToEdgeLengthSquareRatio() const {
constexpr double normFactor = 1.0;

const double a = MathUtils<double>::Norm3(this->GetPoint(0)-this->GetPoint(1));
const double b = MathUtils<double>::Norm3(this->GetPoint(1)-this->GetPoint(2));
const double c = MathUtils<double>::Norm3(this->GetPoint(2)-this->GetPoint(0));

return normFactor * Area() / std::pow(a+b+c, 2);
}


virtual double ShortestAltitudeToLongestEdge() const
{
constexpr double normFactor = 1.0;

const array_1d<double, 3> a = this->GetPoint(0) - this->GetPoint(1);
const array_1d<double, 3> b = this->GetPoint(1) - this->GetPoint(2);
const array_1d<double, 3> c = this->GetPoint(2) - this->GetPoint(0);

const double sa = (a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]);
const double sb = (b[0]*b[0])+(b[1]*b[1])+(b[2]*b[2]);
const double sc = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2]);

const double base = CalculateMaxEdgeLength(sa,sb,sc);

return normFactor * (Area() * 2 / base ) / base;
}


array_1d<double, 3> Normal(const CoordinatesArrayType& rPointLocalCoordinates) const override
{
const array_1d<double, 3> tangent_xi  = this->GetPoint(1) - this->GetPoint(0);
const array_1d<double, 3> tangent_eta = this->GetPoint(2) - this->GetPoint(0);

array_1d<double, 3> normal;
MathUtils<double>::CrossProduct(normal, tangent_xi, tangent_eta);

return 0.5 * normal;
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
const auto center = this->Center();
const array_1d<double, 3> normal = this->UnitNormal(center);

const Point point_to_project(rPoint);
double distance;
CoordinatesArrayType point_projected;
point_projected = GeometricalProjectionUtilities::FastProject( center, point_to_project, normal, distance);

if (std::abs(distance) > std::numeric_limits<double>::epsilon()) {
if (std::abs(distance) > 1.0e-6 * Length()) {
KRATOS_WARNING_FIRST_N("Triangle3D3", 10) << "The " << rPoint << " is in a distance: " << std::abs(distance) << std::endl;
return false;
}

noalias(point_projected) = rPoint - normal * distance;
}

PointLocalCoordinates( rResult, point_projected );

if ( (rResult[0] >= (0.0-Tolerance)) && (rResult[0] <= (1.0+Tolerance)) ) {
if ( (rResult[1] >= (0.0-Tolerance)) && (rResult[1] <= (1.0+Tolerance)) ) {
if ( (rResult[0] + rResult[1]) <= (1.0+Tolerance) ) {
return true;
}
}
}

return false;
}


CoordinatesArrayType& PointLocalCoordinates(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint
) const override
{
noalias(rResult) = ZeroVector(3);

array_1d<double, 3> tangent_xi  = this->GetPoint(1) - this->GetPoint(0);
tangent_xi /= norm_2(tangent_xi);
array_1d<double, 3> tangent_eta = this->GetPoint(2) - this->GetPoint(0);
tangent_eta /= norm_2(tangent_eta);

const auto center = this->Center();

BoundedMatrix<double, 3, 3> rotation_matrix = ZeroMatrix(3, 3);
for (IndexType i = 0; i < 3; ++i) {
rotation_matrix(0, i) = tangent_xi[i];
rotation_matrix(1, i) = tangent_eta[i];
}

CoordinatesArrayType aux_point_to_rotate, destination_point_rotated;
noalias(aux_point_to_rotate) = rPoint - center.Coordinates();
noalias(destination_point_rotated) = prod(rotation_matrix, aux_point_to_rotate) + center.Coordinates();

array_1d<CoordinatesArrayType, 3> points_rotated;
for (IndexType i = 0; i < 3; ++i) {
noalias(aux_point_to_rotate) = this->GetPoint(i).Coordinates() - center.Coordinates();
noalias(points_rotated[i]) = prod(rotation_matrix, aux_point_to_rotate) + center.Coordinates();
}

BoundedMatrix<double, 2, 2> J;
J(0,0) = points_rotated[1][0] - points_rotated[0][0];
J(0,1) = points_rotated[2][0] - points_rotated[0][0];
J(1,0) = points_rotated[1][1] - points_rotated[0][1];
J(1,1) = points_rotated[2][1] - points_rotated[0][1];
const double det_J = J(0,0)*J(1,1) - J(0,1)*J(1,0);

const double eta = (J(1,0)*(points_rotated[0][0] - destination_point_rotated[0]) +
J(0,0)*(destination_point_rotated[1] - points_rotated[0][1])) / det_J;
const double xi  = (J(1,1)*(destination_point_rotated[0] - points_rotated[0][0]) +
J(0,1)*(points_rotated[0][1] - destination_point_rotated[1])) / det_J;

rResult(0) = xi;
rResult(1) = eta;

return rResult;
}



double CalculateDistance(
const CoordinatesArrayType& rPointGlobalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
const Point point(rPointGlobalCoordinates);
return GeometryUtils::PointDistanceToTriangle3D(this->GetPoint(0), this->GetPoint(1), this->GetPoint(2), point);
}



JacobiansType& Jacobian( JacobiansType& rResult,
IntegrationMethod ThisMethod ) const override
{
Matrix jacobian( 3, 2 );
jacobian( 0, 0 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 1 ).X() ); 
jacobian( 1, 0 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 1 ).Y() );
jacobian( 2, 0 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 1 ).Z() );
jacobian( 0, 1 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 2 ).X() );
jacobian( 1, 1 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 2 ).Y() );
jacobian( 2, 1 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 2 ).Z() );

if ( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( this->IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

std::fill( rResult.begin(), rResult.end(), jacobian );

return rResult;
}


JacobiansType& Jacobian( JacobiansType& rResult,
IntegrationMethod ThisMethod,
Matrix & DeltaPosition ) const override
{
Matrix jacobian( 3, 2 );
jacobian( 0, 0 ) = -( BaseType::GetPoint( 0 ).X() - DeltaPosition(0,0) ) + ( BaseType::GetPoint( 1 ).X() - DeltaPosition(1,0) ); 
jacobian( 1, 0 ) = -( BaseType::GetPoint( 0 ).Y() - DeltaPosition(0,1) ) + ( BaseType::GetPoint( 1 ).Y() - DeltaPosition(1,1) );
jacobian( 2, 0 ) = -( BaseType::GetPoint( 0 ).Z() - DeltaPosition(0,2) ) + ( BaseType::GetPoint( 1 ).Z() - DeltaPosition(1,2) );
jacobian( 0, 1 ) = -( BaseType::GetPoint( 0 ).X() - DeltaPosition(0,0) ) + ( BaseType::GetPoint( 2 ).X() - DeltaPosition(2,0) );
jacobian( 1, 1 ) = -( BaseType::GetPoint( 0 ).Y() - DeltaPosition(0,1) ) + ( BaseType::GetPoint( 2 ).Y() - DeltaPosition(2,1) );
jacobian( 2, 1 ) = -( BaseType::GetPoint( 0 ).Z() - DeltaPosition(0,2) ) + ( BaseType::GetPoint( 2 ).Z() - DeltaPosition(2,2) );

if ( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( this->IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

std::fill( rResult.begin(), rResult.end(), jacobian );

return rResult;
}



Matrix& Jacobian( Matrix& rResult,
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod ) const override
{
rResult.resize( 3, 2,false );
rResult( 0, 0 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 1 ).X() ); 
rResult( 1, 0 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 1 ).Y() );
rResult( 2, 0 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 1 ).Z() );
rResult( 0, 1 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 2 ).X() );
rResult( 1, 1 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 2 ).Y() );
rResult( 2, 1 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 2 ).Z() );
return rResult;
}



Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
rResult.resize( 3, 2 ,false);
rResult( 0, 0 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 1 ).X() );
rResult( 1, 0 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 1 ).Y() );
rResult( 2, 0 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 1 ).Z() );
rResult( 0, 1 ) = -( BaseType::GetPoint( 0 ).X() ) + ( BaseType::GetPoint( 2 ).X() );
rResult( 1, 1 ) = -( BaseType::GetPoint( 0 ).Y() ) + ( BaseType::GetPoint( 2 ).Y() );
rResult( 2, 1 ) = -( BaseType::GetPoint( 0 ).Z() ) + ( BaseType::GetPoint( 2 ).Z() );
return rResult;
}


Vector& DeterminantOfJacobian( Vector& rResult, IntegrationMethod ThisMethod ) const override
{
const unsigned int integration_points_number = msGeometryData.IntegrationPointsNumber( ThisMethod );
if(rResult.size() != integration_points_number)
{
rResult.resize(integration_points_number,false);
}

const double detJ = 2.0*(this->Area());

for ( unsigned int pnt = 0; pnt < integration_points_number; pnt++ )
{
rResult[pnt] = detJ;
}
return rResult;
}


double DeterminantOfJacobian( IndexType IntegrationPoint,
IntegrationMethod ThisMethod ) const override
{
return 2.0*(this->Area());
}


double DeterminantOfJacobian( const CoordinatesArrayType& rPoint ) const override
{
return 2.0*(this->Area());
}



SizeType EdgesNumber() const override
{
return 3;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 1 ), this->pGetPoint( 2 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 2 ), this->pGetPoint( 0 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ) ) );
return edges;
}



SizeType FacesNumber() const override
{
return 1;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();

faces.push_back( Kratos::make_shared<FaceType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ), this->pGetPoint( 2 )) );
return faces;
}

void NumberNodesInFaces (DenseVector<unsigned int>& NumberNodesInFaces) const override
{
if(NumberNodesInFaces.size() != 3 )
NumberNodesInFaces.resize(3,false);
NumberNodesInFaces[0]=2;
NumberNodesInFaces[1]=2;
NumberNodesInFaces[2]=2;

}

void NodesInFaces (DenseMatrix<unsigned int>& NodesInFaces) const override
{
if(NodesInFaces.size1() != 3 || NodesInFaces.size2() != 3)
NodesInFaces.resize(3,3,false);

NodesInFaces(0,0)=0;
NodesInFaces(1,0)=1;
NodesInFaces(2,0)=2;
NodesInFaces(0,1)=1;
NodesInFaces(1,1)=2;
NodesInFaces(2,1)=0;
NodesInFaces(0,2)=2;
NodesInFaces(1,2)=0;
NodesInFaces(2,2)=1;

}



GeometriesArrayType Faces( void ) override
{
return GeometriesArrayType();
}




double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return( 1.0 -rPoint[0] - rPoint[1] );
case 1:
return( rPoint[0] );
case 2:
return( rPoint[1] );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}



Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 3)
{
rResult.resize(3,false);
}

rResult[0] =  1.0 -rCoordinates[0] - rCoordinates[1];
rResult[1] =  rCoordinates[0] ;
rResult[2] =  rCoordinates[1] ;

return rResult;
}



std::string Info() const override
{
return "2 dimensional triangle with three nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "2 dimensional triangle with three nodes in 3D space";
}



void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
Jacobian( jacobian, PointType() );
rOStream << "    Jacobian in the origin\t : " << jacobian;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients(
IntegrationMethod ThisMethod )
{
ShapeFunctionsGradientsType localGradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients( ThisMethod );
const int integration_points_number
= msGeometryData.IntegrationPointsNumber( ThisMethod );
ShapeFunctionsGradientsType Result( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Result[pnt] = localGradients[pnt];
}

return Result;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients()
{
IntegrationMethod ThisMethod = msGeometryData.DefaultIntegrationMethod();
ShapeFunctionsGradientsType localGradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients( ThisMethod );
const int integration_points_number
= msGeometryData.IntegrationPointsNumber( ThisMethod );
ShapeFunctionsGradientsType Result( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Result[pnt] = localGradients[pnt];
}

return Result;
}


Matrix& ShapeFunctionsLocalGradients( Matrix& rResult,
const CoordinatesArrayType& rPoint ) const override
{
rResult.resize( 3, 2 ,false);
noalias( rResult ) = ZeroMatrix( 3, 2 );
rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) =  0.0;
rResult( 2, 0 ) =  0.0;
rResult( 2, 1 ) =  1.0;
return rResult;
}




virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, PointType& rPoint )
{
rResult.resize( 3, 2,false );
noalias( rResult ) = ZeroMatrix( 3, 2 );
rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) =  0.0;
rResult( 2, 0 ) =  0.0;
rResult( 2, 1 ) =  1.0;
return rResult;
}


ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives( ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsGradientsType temp( this->PointsNumber() );
rResult.swap( temp );
}

rResult[0].resize( 2, 2 ,false);

rResult[1].resize( 2, 2 ,false);
rResult[2].resize( 2, 2 ,false);
rResult[0]( 0, 0 ) = 0.0;
rResult[0]( 0, 1 ) = 0.0;
rResult[0]( 1, 0 ) = 0.0;
rResult[0]( 1, 1 ) = 0.0;
rResult[1]( 0, 0 ) = 0.0;
rResult[1]( 0, 1 ) = 0.0;
rResult[1]( 1, 0 ) = 0.0;
rResult[1]( 1, 1 ) = 0.0;
rResult[2]( 0, 0 ) = 0.0;
rResult[2]( 0, 1 ) = 0.0;
rResult[2]( 1, 0 ) = 0.0;
rResult[2]( 1, 1 ) = 0.0;
return rResult;
}


ShapeFunctionsThirdDerivativesType& ShapeFunctionsThirdDerivatives( ShapeFunctionsThirdDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsThirdDerivativesType temp( this->PointsNumber() );
rResult.swap( temp );
}

for ( IndexType i = 0; i < rResult.size(); i++ )
{
DenseVector<Matrix> temp( this->PointsNumber() );
rResult[i].swap( temp );
}

rResult[0][0].resize( 2, 2 ,false);

rResult[0][1].resize( 2, 2 ,false);
rResult[1][0].resize( 2, 2 ,false);
rResult[1][1].resize( 2, 2 ,false);
rResult[2][0].resize( 2, 2 ,false);
rResult[2][1].resize( 2, 2 ,false);

for ( int i = 0; i < 3; i++ )
{
rResult[i][0]( 0, 0 ) = 0.0;
rResult[i][0]( 0, 1 ) = 0.0;
rResult[i][0]( 1, 0 ) = 0.0;
rResult[i][0]( 1, 1 ) = 0.0;
rResult[i][1]( 0, 0 ) = 0.0;
rResult[i][1]( 0, 1 ) = 0.0;
rResult[i][1]( 1, 0 ) = 0.0;
rResult[i][1]( 1, 1 ) = 0.0;
}

return rResult;
}



KRATOS_DEPRECATED_MESSAGE("This method is deprecated. Use either \'ProjectionPointLocalToLocalSpace\' or \'ProjectionPointGlobalToLocalSpace\' instead.")
int ProjectionPoint(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectedPointGlobalCoordinates,
CoordinatesArrayType& rProjectedPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
KRATOS_WARNING("ProjectionPoint") << "This method is deprecated. Use either \'ProjectionPointLocalToLocalSpace\' or \'ProjectionPointGlobalToLocalSpace\' instead." << std::endl;

ProjectionPointGlobalToLocalSpace(rPointGlobalCoordinates, rProjectedPointLocalCoordinates, Tolerance);

this->GlobalCoordinates(rProjectedPointGlobalCoordinates, rProjectedPointLocalCoordinates);

return 1;
}

int ProjectionPointLocalToLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
for (std::size_t  i = 0; i < 3; ++i) {
rProjectionPointLocalCoordinates[i] = (rPointLocalCoordinates[i] < 0.0) ? 0.0 : rPointLocalCoordinates[i];
rProjectionPointLocalCoordinates[i] = (rPointLocalCoordinates[i] > 1.0) ? 1.0 : rPointLocalCoordinates[i];
}

return 1;
}

int ProjectionPointGlobalToLocalSpace(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
PointLocalCoordinates(rProjectionPointLocalCoordinates, rPointGlobalCoordinates);

CoordinatesArrayType point_local_coordinates(rProjectionPointLocalCoordinates);
return ProjectionPointLocalToLocalSpace(point_local_coordinates, rProjectionPointLocalCoordinates);
}



protected:


private:

static const GeometryData msGeometryData;

static const GeometryDimension msGeometryDimension;


friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );
}

Triangle3D3(): BaseType( PointsArrayType(), &msGeometryData ) {}







static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 3;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
row( shape_function_values, pnt )[0] = 1.0
- integration_points[pnt].X()
- integration_points[pnt].Y();
row( shape_function_values, pnt )[1] = integration_points[pnt].X();
row( shape_function_values, pnt )[2] = integration_points[pnt].Y();
}

return shape_function_values;
}


static ShapeFunctionsGradientsType
CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Matrix result( 3, 2 );
result( 0, 0 ) = -1.0;
result( 0, 1 ) = -1.0;
result( 1, 0 ) =  1.0;
result( 1, 1 ) =  0.0;
result( 2, 0 ) =  0.0;
result( 2, 1 ) =  1.0;
d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}


static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature<TriangleGaussLegendreIntegrationPoints1, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints2, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints3, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints4, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints5, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleCollocationIntegrationPoints1, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleCollocationIntegrationPoints2, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleCollocationIntegrationPoints3, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleCollocationIntegrationPoints4, 2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<TriangleCollocationIntegrationPoints5, 2, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}


static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_values;
}


static const ShapeFunctionsLocalGradientsContainerType
AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients =
{
{
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Triangle3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}


inline double CalculateMinEdgeLength(const double a, const double b, const double c) const {
return std::sqrt(std::min({a, b, c}));
}


inline double CalculateMaxEdgeLength(const double a, const double b, const double c) const {
return std::sqrt(std::max({a, b, c}));
}


inline double CalculateAvgEdgeLength(const double a, const double b, const double c) const {
constexpr double onethird = 1.0 / 3.0;
return (a+b+c) * onethird;
}


inline double CalculateCircumradius(const double a, const double b, const double c) const {
return (a*b*c) / std::sqrt((a+b+c) * (b+c-a) * (c+a-b) * (a+b-c));
}


inline double CalculateInradius(const double a, const double b, const double c) const {
return 0.5 * std::sqrt((b+c-a) * (c+a-b) * (a+b-c) / (a+b+c));
}

bool LineTriangleOverlap(
const Point& rPoint1,
const Point& rPoint2) const
{
array_1d<double,3> intersection_point;
const int result = IntersectionUtilities::ComputeTriangleLineIntersection(*this, rPoint1, rPoint2, intersection_point);
return result == 1 ? true : false;
}

bool TriangleTriangleOverlap(
const Point& rPoint1,
const Point& rPoint2,
const Point& rPoint3) const
{

Plane3D plane_1(this->GetPoint(0), this->GetPoint(1), this->GetPoint(2));
array_1d<double, 3> distances_1;
distances_1[0] = plane_1.CalculateSignedDistance(rPoint1);
distances_1[1] = plane_1.CalculateSignedDistance(rPoint2);
distances_1[2] = plane_1.CalculateSignedDistance(rPoint3);
if (AllSameSide(distances_1))
return false;

Plane3D plane_2(rPoint1, rPoint2, rPoint3);
array_1d<double, 3> distances_2;
for (int i = 0; i < 3; ++i)
distances_2[i] = plane_2.CalculateSignedDistance(this->GetPoint(i));
if (AllSameSide(distances_2))
return false;

array_1d<double, 3> intersection_direction;
MathUtils<double>::CrossProduct(intersection_direction, plane_1.GetNormal(), plane_2.GetNormal());

int index = GetMajorAxis(intersection_direction);

double vp0 = this->GetPoint(0)[index];
double vp1 = this->GetPoint(1)[index];
double vp2 = this->GetPoint(2)[index];

double up0 = rPoint1[index];
double up1 = rPoint2[index];
double up2 = rPoint3[index];

double a, b, c, x0, x1;
if (ComputeIntervals(vp0, vp1, vp2, distances_2[0], distances_2[1], distances_2[2], a, b, c, x0, x1))
{
return CoplanarIntersectionCheck(plane_1.GetNormal(), rPoint1, rPoint2, rPoint3);
}

double d, e, f, y0, y1;
if (ComputeIntervals(up0, up1, up2, distances_1[0], distances_1[1], distances_1[2], d, e, f, y0, y1))

{
return CoplanarIntersectionCheck(plane_1.GetNormal(), rPoint1, rPoint2, rPoint3);
}

double xx, yy, xxyy, tmp;
xx = x0*x1;
yy = y0*y1;
xxyy = xx*yy;

array_1d<double, 2> isect1, isect2;

tmp = a*xxyy;
isect1[0] = tmp + b*x1*yy;
isect1[1] = tmp + c*x0*yy;

tmp = d*xxyy;
isect2[0] = tmp + e*xx*y1;
isect2[1] = tmp + f*xx*y0;

if (isect1[0] > isect1[1]) {
isect1[1] = isect1[0] + isect1[1];
isect1[0] = isect1[1] - isect1[0];
isect1[1] = isect1[1] - isect1[0];
}

if (isect2[0] > isect2[1]) {
isect2[1] = isect2[0] + isect2[1];
isect2[0] = isect2[1] - isect2[0];
isect2[1] = isect2[1] - isect2[0];
}

return (isect1[1]<isect2[0] || isect2[1]<isect1[0]) ? false : true;
}

bool ComputeIntervals(
double& VV0,
double& VV1,
double& VV2,
double& D0,
double& D1,
double& D2,
double& A,
double& B,
double& C,
double& X0,
double& X1
) const
{
double D0D1 = D0 * D1;
double D0D2 = D0 * D2;

if (D0D1 > 0.0) {
A = VV2;
B = (VV0 - VV2)*D2;
C = (VV1 - VV2)*D2;
X0 = D2 - D0;
X1 = D2 - D1;
}
else if (D0D2 > 0.0) {
A = VV1;
B = (VV0 - VV1)*D1;
C = (VV2 - VV1)*D1;
X0 = D1 - D0;
X1 = D1 - D2;
}
else if (D1 * D2 > 0.00 || D0 != 0.00) {
A = VV0;
B = (VV1 - VV0)*D0;
C = (VV2 - VV0)*D0;
X0 = D0 - D1;
X1 = D0 - D2;
}
else if (D1 != 0.00) {
A = VV1;
B = (VV0 - VV1)*D1;
C = (VV2 - VV1)*D1;
X0 = D1 - D0;
X1 = D1 - D2;
}
else if (D2 != 0.00) {
A = VV2;
B = (VV0 - VV2)*D2;
C = (VV1 - VV2)*D2;
X0 = D2 - D0;
X1 = D2 - D1;
}
else { 
return true;
}
return false;
}

bool CoplanarIntersectionCheck(
const array_1d<double,3>& N,
const Point& rPoint1,
const Point& rPoint2,
const Point& rPoint3) const
{
array_1d<double, 3 > A;
int i0, i1;

A[0] = std::abs(N[0]);
A[1] = std::abs(N[1]);
A[2] = std::abs(N[2]);
if (A[0] > A[1]) {
if (A[0] > A[2]) {
i0 = 1;      
i1 = 2;
} else {
i0 = 0;      
i1 = 1;
}
} else {             
if (A[2] > A[1]) {
i0 = 0;      
i1 = 1;
} else {
i0 = 0;      
i1 = 2;
}
}

if (EdgeToTriangleEdgesCheck(i0, i1, this->GetPoint(0), this->GetPoint(1), rPoint1, rPoint2, rPoint3)) return true;
if (EdgeToTriangleEdgesCheck(i0, i1, this->GetPoint(1), this->GetPoint(2), rPoint1, rPoint2, rPoint3)) return true;
if (EdgeToTriangleEdgesCheck(i0, i1, this->GetPoint(2), this->GetPoint(0), rPoint1, rPoint2, rPoint3)) return true;


if (PointInTriangle(i0, i1, this->GetPoint(0), rPoint1, rPoint2, rPoint3)) return true;
else if (PointInTriangle(i0, i1, rPoint1, this->GetPoint(0), this->GetPoint(1), this->GetPoint(2))) return true;

return false;
}

bool EdgeToTriangleEdgesCheck(
const int& i0,
const int& i1,
const Point& V0,
const Point& V1,
const Point&U0,
const Point&U1,
const Point&U2) const
{
double Ax, Ay, Bx, By, Cx, Cy, e, d, f;
Ax = V1[i0] - V0[i0];
Ay = V1[i1] - V0[i1];

if (EdgeToEdgeIntersectionCheck(Ax, Ay, Bx, By, Cx, Cy, e, d, f, i0, i1, V0, U0, U1) == true) return true;

if (EdgeToEdgeIntersectionCheck(Ax, Ay, Bx, By, Cx, Cy, e, d, f, i0, i1, V0, U1, U2) == true) return true;

if (EdgeToEdgeIntersectionCheck(Ax, Ay, Bx, By, Cx, Cy, e, d, f, i0, i1, V0, U2, U0) == true) return true;

return false;
}

bool EdgeToEdgeIntersectionCheck(
double& Ax,
double& Ay,
double& Bx,
double& By,
double& Cx,
double& Cy,
double& e,
double& d,
double& f,
const int& i0,
const int& i1,
const Point& V0,
const Point& U0,
const Point& U1) const
{
Bx = U0[i0] - U1[i0];
By = U0[i1] - U1[i1];
Cx = V0[i0] - U0[i0];
Cy = V0[i1] - U0[i1];
f = Ay*Bx - Ax*By;
d = By*Cx - Bx*Cy;

if (std::abs(f) < 1E-10) f = 0.00;
if (std::abs(d) < 1E-10) d = 0.00;

if ((f>0.00 && d >= 0.00 && d <= f) || (f<0.00 && d <= 0.00 && d >= f)) {
e = Ax*Cy - Ay*Cx;

if (f > 0.0) {
if (e >= 0.0 && e <= f) return true;
} else {
if (e <= 0.0 && e >= f) return true;
}
}
return false;
}

bool PointInTriangle(
int i0,
int i1,
const Point& V0,
const Point& U0,
const Point& U1,
const Point& U2) const
{
double a,b,c,d0,d1,d2;


a =   U1[i1] - U0[i1];
b = -(U1[i0] - U0[i0]);
c = -a * U0[i0] -b * U0[i1];
d0=  a * V0[i0] +b * V0[i1] + c;

a =   U2[i1] - U1[i1];
b = -(U2[i0] - U1[i0]);
c = -a * U1[i0] -b * U1[i1];
d1=  a * V0[i0] +b * V0[i1] + c;

a =   U0[i1] - U2[i1];
b = -(U0[i0] - U2[i0]);
c = -a * U2[i0] - b * U2[i1];
d2 = a * V0[i0] + b * V0[i1] + c;

if (d0 * d1 > 0.0){
if (d0 * d2 > 0.0) return true;
}
return false;
}


inline bool TriBoxOverlap(Point& rBoxCenter, Point& rBoxHalfSize) const
{
double abs_ex, abs_ey, abs_ez, distance;
array_1d<double,3 > vert0, vert1, vert2;
array_1d<double,3 > edge0, edge1, edge2, normal;
std::pair<double, double> min_max;

noalias(vert0) = this->GetPoint(0) - rBoxCenter;
noalias(vert1) = this->GetPoint(1) - rBoxCenter;
noalias(vert2) = this->GetPoint(2) - rBoxCenter;

noalias(edge0) = vert1 - vert0;
noalias(edge1) = vert2 - vert1;
noalias(edge2) = vert0 - vert2;

abs_ex = std::abs(edge0[0]);
abs_ey = std::abs(edge0[1]);
abs_ez = std::abs(edge0[2]);
if (AxisTestX(edge0[1],edge0[2],abs_ey,abs_ez,vert0,vert2,rBoxHalfSize)) return false;
if (AxisTestY(edge0[0],edge0[2],abs_ex,abs_ez,vert0,vert2,rBoxHalfSize)) return false;
if (AxisTestZ(edge0[0],edge0[1],abs_ex,abs_ey,vert0,vert2,rBoxHalfSize)) return false;

abs_ex = std::abs(edge1[0]);
abs_ey = std::abs(edge1[1]);
abs_ez = std::abs(edge1[2]);
if (AxisTestX(edge1[1],edge1[2],abs_ey,abs_ez,vert1,vert0,rBoxHalfSize)) return false;
if (AxisTestY(edge1[0],edge1[2],abs_ex,abs_ez,vert1,vert0,rBoxHalfSize)) return false;
if (AxisTestZ(edge1[0],edge1[1],abs_ex,abs_ey,vert1,vert0,rBoxHalfSize)) return false;

abs_ex = std::abs(edge2[0]);
abs_ey = std::abs(edge2[1]);
abs_ez = std::abs(edge2[2]);
if (AxisTestX(edge2[1],edge2[2],abs_ey,abs_ez,vert2,vert1,rBoxHalfSize)) return false;
if (AxisTestY(edge2[0],edge2[2],abs_ex,abs_ez,vert2,vert1,rBoxHalfSize)) return false;
if (AxisTestZ(edge2[0],edge2[1],abs_ex,abs_ey,vert2,vert1,rBoxHalfSize)) return false;


min_max = std::minmax({vert0[0], vert1[0], vert2[0]});
if(min_max.first>rBoxHalfSize[0] || min_max.second<-rBoxHalfSize[0]) return false;

min_max = std::minmax({vert0[1], vert1[1], vert2[1]});
if(min_max.first>rBoxHalfSize[1] || min_max.second<-rBoxHalfSize[1]) return false;

min_max = std::minmax({vert0[2], vert1[2], vert2[2]});
if(min_max.first>rBoxHalfSize[2] || min_max.second<-rBoxHalfSize[2]) return false;

MathUtils<double>::CrossProduct(normal, edge0, edge1);
distance = -inner_prod(normal, vert0);
if(!PlaneBoxOverlap(normal, distance, rBoxHalfSize)) return false;

return true;  
}


bool PlaneBoxOverlap(const array_1d<double,3>& rNormal, const double& rDist, const array_1d<double,3>& rMaxBox) const
{
array_1d<double,3> vmin, vmax;
for(int q = 0; q < 3; q++)
{
if(rNormal[q] > 0.00)
{
vmin[q] = -rMaxBox[q];
vmax[q] =  rMaxBox[q];
}
else
{
vmin[q] =  rMaxBox[q];
vmax[q] = -rMaxBox[q];
}
}
if(inner_prod(rNormal, vmin) + rDist >  0.00) return false;
if(inner_prod(rNormal, vmax) + rDist >= 0.00) return true;

return false;
}


bool AxisTestX(double& rEdgeY, double& rEdgeZ,
double& rAbsEdgeY, double& rAbsEdgeZ,
array_1d<double,3>& rVertA,
array_1d<double,3>& rVertC,
Point& rBoxHalfSize) const
{
double proj_a, proj_c, rad;
proj_a = rEdgeY*rVertA[2] - rEdgeZ*rVertA[1];
proj_c = rEdgeY*rVertC[2] - rEdgeZ*rVertC[1];
std::pair<double, double> min_max = std::minmax(proj_a, proj_c);

rad = rAbsEdgeZ*rBoxHalfSize[1] + rAbsEdgeY*rBoxHalfSize[2];

if(min_max.first>rad || min_max.second<-rad) return true;
else return false;
}


bool AxisTestY(double& rEdgeX, double& rEdgeZ,
double& rAbsEdgeX, double& rAbsEdgeZ,
array_1d<double,3>& rVertA,
array_1d<double,3>& rVertC,
Point& rBoxHalfSize) const
{
double proj_a, proj_c, rad;
proj_a = rEdgeZ*rVertA[0] - rEdgeX*rVertA[2];
proj_c = rEdgeZ*rVertC[0] - rEdgeX*rVertC[2];
std::pair<double, double> min_max = std::minmax(proj_a, proj_c);

rad = rAbsEdgeZ*rBoxHalfSize[0] + rAbsEdgeX*rBoxHalfSize[2];

if(min_max.first>rad || min_max.second<-rad) return true;
else return false;
}


bool AxisTestZ(double& rEdgeX, double& rEdgeY,
double& rAbsEdgeX, double& rAbsEdgeY,
array_1d<double,3>& rVertA,
array_1d<double,3>& rVertC,
Point& rBoxHalfSize) const
{
double proj_a, proj_c, rad;
proj_a = rEdgeX*rVertA[1] - rEdgeY*rVertA[0];
proj_c = rEdgeX*rVertC[1] - rEdgeY*rVertC[0];
std::pair<double, double> min_max = std::minmax(proj_a, proj_c);

rad = rAbsEdgeY*rBoxHalfSize[0] + rAbsEdgeX*rBoxHalfSize[1];

if(min_max.first>rad || min_max.second<-rad) return true;
else return false;
}






template<class TOtherPointType> friend class Triangle3D3;




}; 




template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream,
Triangle3D3<TPointType>& rThis );

template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const Triangle3D3<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}


template<class TPointType> const
GeometryData Triangle3D3<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
Triangle3D3<TPointType>::AllIntegrationPoints(),
Triangle3D3<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType>
const GeometryDimension Triangle3D3<TPointType>::msGeometryDimension(3, 2);

}