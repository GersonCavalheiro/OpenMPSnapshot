
#pragma once



#include "geometries/quadrilateral_3d_4.h"
#include "utilities/integration_utilities.h"
#include "integration/hexahedron_gauss_legendre_integration_points.h"
#include "integration/hexahedron_gauss_lobatto_integration_points.h"

namespace Kratos
{

template<class TPointType> class Hexahedra3D8 : public Geometry<TPointType>
{
public:



typedef Geometry<TPointType> BaseType;


typedef Line3D2<TPointType> EdgeType;
typedef Quadrilateral3D4<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Hexahedra3D8 );


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef typename BaseType::GeometriesArrayType GeometriesArrayType;


typedef TPointType PointType;


typedef typename BaseType::IndexType IndexType;



typedef typename BaseType::SizeType SizeType;


typedef typename BaseType::PointsArrayType PointsArrayType;


typedef typename BaseType::IntegrationPointType IntegrationPointType;


typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


typedef typename BaseType::IntegrationPointsContainerType IntegrationPointsContainerType;


typedef typename BaseType::ShapeFunctionsValuesContainerType
ShapeFunctionsValuesContainerType;


typedef typename BaseType::ShapeFunctionsLocalGradientsContainerType
ShapeFunctionsLocalGradientsContainerType;


typedef typename BaseType::JacobiansType JacobiansType;


typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef typename BaseType::ShapeFunctionsSecondDerivativesType ShapeFunctionsSecondDerivativesType;


typedef typename BaseType::NormalType NormalType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;


typedef Matrix MatrixType;



Hexahedra3D8( const PointType& Point1, const PointType& Point2,
const PointType& Point3, const PointType& Point4,
const PointType& Point5, const PointType& Point6,
const PointType& Point7, const PointType& Point8 )
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( typename PointType::Pointer( new PointType( Point1 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point2 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point3 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point4 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point5 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point6 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point7 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point8 ) ) );
}

Hexahedra3D8( typename PointType::Pointer pPoint1,
typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3,
typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5,
typename PointType::Pointer pPoint6,
typename PointType::Pointer pPoint7,
typename PointType::Pointer pPoint8 )
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( pPoint1 );
this->Points().push_back( pPoint2 );
this->Points().push_back( pPoint3 );
this->Points().push_back( pPoint4 );
this->Points().push_back( pPoint5 );
this->Points().push_back( pPoint6 );
this->Points().push_back( pPoint7 );
this->Points().push_back( pPoint8 );
}

explicit Hexahedra3D8( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 8 )
KRATOS_ERROR << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D8(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 8 ) << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D8(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType( rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 8) << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}


Hexahedra3D8( Hexahedra3D8 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> explicit Hexahedra3D8( Hexahedra3D8<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Hexahedra3D8() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Hexahedra;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Hexahedra3D8;
}




Hexahedra3D8& operator=( const Hexahedra3D8& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Hexahedra3D8& operator=( Hexahedra3D8<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}





typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Hexahedra3D8( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Hexahedra3D8( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}




double Length() const override
{
return std::sqrt( std::abs( this->DeterminantOfJacobian( PointType() ) ) );
}


double Area() const override
{
return Volume();
}


double Volume() const override
{
return IntegrationUtilities::ComputeVolume3DGeometry(*this);
}


double DomainSize() const override
{
return Volume();
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
if ( rResult.size1() != 8 || rResult.size2() != 3 )
rResult.resize( 8, 3, false );

rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 0, 2 ) = -1.0;

rResult( 1, 0 ) = 1.0;
rResult( 1, 1 ) = -1.0;
rResult( 1, 2 ) = -1.0;

rResult( 2, 0 ) = 1.0;
rResult( 2, 1 ) = 1.0;
rResult( 2, 2 ) = -1.0;

rResult( 3, 0 ) = -1.0;
rResult( 3, 1 ) = 1.0;
rResult( 3, 2 ) = -1.0;

rResult( 4, 0 ) = -1.0;
rResult( 4, 1 ) = -1.0;
rResult( 4, 2 ) = 1.0;

rResult( 5, 0 ) = 1.0;
rResult( 5, 1 ) = -1.0;
rResult( 5, 2 ) = 1.0;

rResult( 6, 0 ) = 1.0;
rResult( 6, 1 ) = 1.0;
rResult( 6, 2 ) = 1.0;

rResult( 7, 0 ) = -1.0;
rResult( 7, 1 ) = 1.0;
rResult( 7, 2 ) = 1.0;

return rResult;
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) ) {
if ( std::abs( rResult[1] ) <= (1.0 + Tolerance) ) {
if ( std::abs( rResult[2] ) <= (1.0 + Tolerance) ) {
return true;
}
}
}

return false;
}



SizeType EdgesNumber() const override
{
return 12;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer EdgePointerType;
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 0 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 5 ),
this->pGetPoint( 6 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 6 ),
this->pGetPoint( 7 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 7 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 5 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 7 ) ) ) );
return edges;
}


double AverageEdgeLength() const override
{
const TPointType& p0 = this->GetPoint(0);
const TPointType& p1 = this->GetPoint(1);
const TPointType& p2 = this->GetPoint(2);
const TPointType& p3 = this->GetPoint(3);
const TPointType& p4 = this->GetPoint(4);
const TPointType& p5 = this->GetPoint(5);
const TPointType& p6 = this->GetPoint(6);
const TPointType& p7 = this->GetPoint(7);
return (MathUtils<double>::Norm3(p0-p1) +
MathUtils<double>::Norm3(p1-p2) +
MathUtils<double>::Norm3(p2-p3) +
MathUtils<double>::Norm3(p3-p0) +
MathUtils<double>::Norm3(p4-p5) +
MathUtils<double>::Norm3(p5-p6) +
MathUtils<double>::Norm3(p6-p7) +
MathUtils<double>::Norm3(p7-p4) +
MathUtils<double>::Norm3(p0-p4) +
MathUtils<double>::Norm3(p1-p5) +
MathUtils<double>::Norm3(p2-p6) +
MathUtils<double>::Norm3(p3-p7)) /12.0;
}



SizeType FacesNumber() const override
{
return 6;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer FacePointerType;
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 3 ),
this->pGetPoint( 2 ),
this->pGetPoint( 1 ),
this->pGetPoint( 0 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 5 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ),
this->pGetPoint( 5 ),
this->pGetPoint( 1 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 6 ),
this->pGetPoint( 2 ),
this->pGetPoint( 3 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ),
this->pGetPoint( 6 ),
this->pGetPoint( 7 ) ) ) );
return faces;
}


bool HasIntersection( const Point& rLowPoint, const Point& rHighPoint ) const override
{
using Quadrilateral3D4Type = Quadrilateral3D4<TPointType>;
if(Quadrilateral3D4Type(this->pGetPoint(3),this->pGetPoint(2), this->pGetPoint(1), this->pGetPoint(0)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(Quadrilateral3D4Type(this->pGetPoint(0),this->pGetPoint(1), this->pGetPoint(5), this->pGetPoint(4)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(Quadrilateral3D4Type(this->pGetPoint(2),this->pGetPoint(6), this->pGetPoint(5), this->pGetPoint(1)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(Quadrilateral3D4Type(this->pGetPoint(7),this->pGetPoint(6), this->pGetPoint(2), this->pGetPoint(3)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(Quadrilateral3D4Type(this->pGetPoint(7),this->pGetPoint(3), this->pGetPoint(0), this->pGetPoint(4)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(Quadrilateral3D4Type(this->pGetPoint(4),this->pGetPoint(5), this->pGetPoint(6), this->pGetPoint(7)).HasIntersection(rLowPoint, rHighPoint))
return true;

CoordinatesArrayType local_coordinates;
if(IsInside(rLowPoint,local_coordinates))
return true;

return false;
}


void ComputeSolidAngles(Vector& rSolidAngles) const override
{
if(rSolidAngles.size() != 8) {
rSolidAngles.resize(8, false);
}

Vector dihedral_angles(24);
ComputeDihedralAngles(dihedral_angles);

for (unsigned int i = 0; i < 8; ++i) {
rSolidAngles[i] = dihedral_angles[3*i]
+ dihedral_angles[3*i + 1]
+ dihedral_angles[3*i + 2]
- Globals::Pi;
}
}


void ComputeDihedralAngles(Vector& rDihedralAngles) const override
{
if(rDihedralAngles.size() != 24) {
rDihedralAngles.resize(24, false);
}
const auto faces = this->GenerateFaces();
const std::array<unsigned int, 8> faces_0 = {0,0,0,0,5,5,5,5};
const std::array<unsigned int, 8> faces_1 = {1,1,3,3,1,1,3,3};
const std::array<unsigned int, 8> faces_2 = {4,2,2,4,4,2,2,4};

array_1d<double, 3> normal_0, normal_1, normal_2;
double dihedral_angle_0, dihedral_angle_1, dihedral_angle_2;
for (unsigned int i = 0; i < 8; ++i) {
const TPointType& r_point_i = this->GetPoint(i);
noalias(normal_0) = faces[faces_0[i]].UnitNormal(r_point_i);
noalias(normal_1) = faces[faces_1[i]].UnitNormal(r_point_i);
noalias(normal_2) = faces[faces_2[i]].UnitNormal(r_point_i);
dihedral_angle_0 = std::acos(inner_prod(normal_0, -normal_1));
dihedral_angle_1 = std::acos(inner_prod(normal_0, -normal_2));
dihedral_angle_2 = std::acos(inner_prod(normal_2, -normal_1));
rDihedralAngles[i*3] = dihedral_angle_0;
rDihedralAngles[i*3 + 1] = dihedral_angle_1;
rDihedralAngles[i*3 + 2] = dihedral_angle_2;
}
}


double MinDihedralAngle() const override {
Vector dihedral_angles(24);
ComputeDihedralAngles(dihedral_angles);
double min_dihedral_angle = dihedral_angles[0];
for (unsigned int i = 1; i < 24; i++) {
min_dihedral_angle = std::min(dihedral_angles[i], min_dihedral_angle);
}
return min_dihedral_angle;
}


double MaxDihedralAngle() const override {
Vector dihedral_angles(24);
ComputeDihedralAngles(dihedral_angles);
double max_dihedral_angle = dihedral_angles[0];
for (unsigned int i = 1; i < 24; i++) {
max_dihedral_angle = std::max(dihedral_angles[i], max_dihedral_angle);
}
return max_dihedral_angle;
}


double VolumeToRMSEdgeLength() const override {
const auto edges = GenerateEdges();
double sum_squared_lengths = 0.0;
for (const auto& r_edge : edges) {
const double length = r_edge.Length();
sum_squared_lengths += length*length;
}

const double rms_edge = std::sqrt(1.0/12.0 * sum_squared_lengths);

return Volume() / std::pow(rms_edge, 3.0);
}


double ShortestToLongestEdgeQuality() const override {
const auto edges = GenerateEdges();
double min_edge_length = std::numeric_limits<double>::max();
double max_edge_length = -std::numeric_limits<double>::max();
for (const auto& r_edge: edges) {
min_edge_length = std::min(min_edge_length, r_edge.Length());
max_edge_length = std::max(max_edge_length, r_edge.Length());
}
return min_edge_length / max_edge_length;
}




double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return( 0.125*( 1.0 - rPoint[0] )*( 1.0 - rPoint[1] )*( 1.0 - rPoint[2] ) );
case 1:
return( 0.125*( 1.0 + rPoint[0] )*( 1.0 - rPoint[1] )*( 1.0 - rPoint[2] ) );
case 2:
return( 0.125*( 1.0 + rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 - rPoint[2] ) );
case 3:
return( 0.125*( 1.0 - rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 - rPoint[2] ) );
case 4:
return( 0.125*( 1.0 - rPoint[0] )*( 1.0 - rPoint[1] )*( 1.0 + rPoint[2] ) );
case 5:
return( 0.125*( 1.0 + rPoint[0] )*( 1.0 - rPoint[1] )*( 1.0 + rPoint[2] ) );
case 6:
return( 0.125*( 1.0 + rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 + rPoint[2] ) );
case 7:
return( 0.125*( 1.0 - rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 + rPoint[2] ) );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 8) rResult.resize(8,false);
rResult[0] =  0.125*( 1.0 - rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ;
rResult[1] =  0.125*( 1.0 + rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ;
rResult[2] =  0.125*( 1.0 + rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ;
rResult[3] =  0.125*( 1.0 - rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ;
rResult[4] =  0.125*( 1.0 - rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ;
rResult[5] =  0.125*( 1.0 + rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ;
rResult[6] =  0.125*( 1.0 + rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ;
rResult[7] =  0.125*( 1.0 - rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ;
return rResult;
}


Matrix& ShapeFunctionsLocalGradients( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size1() != 8 || rResult.size2() != 3 )
rResult.resize( 8, 3, false );

rResult( 0, 0 ) = -0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 - rPoint[2] );
rResult( 0, 1 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[2] );
rResult( 0, 2 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[1] );
rResult( 1, 0 ) =  0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 - rPoint[2] );
rResult( 1, 1 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[2] );
rResult( 1, 2 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[1] );
rResult( 2, 0 ) =  0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 - rPoint[2] );
rResult( 2, 1 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[2] );
rResult( 2, 2 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] );
rResult( 3, 0 ) = -0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 - rPoint[2] );
rResult( 3, 1 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[2] );
rResult( 3, 2 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[1] );
rResult( 4, 0 ) = -0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 + rPoint[2] );
rResult( 4, 1 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[2] );
rResult( 4, 2 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[1] );
rResult( 5, 0 ) =  0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 + rPoint[2] );
rResult( 5, 1 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] );
rResult( 5, 2 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[1] );
rResult( 6, 0 ) =  0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] );
rResult( 6, 1 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] );
rResult( 6, 2 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] );
rResult( 7, 0 ) = -0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] );
rResult( 7, 1 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[2] );
rResult( 7, 2 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[1] );

return rResult;
}



ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives( ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsGradientsType temp( this->PointsNumber() );
rResult.swap( temp );
}

for ( unsigned int i = 0; i < this->PointsNumber(); ++i )
{
rResult[i].resize(3, 3, false);
}

rResult[0]( 0, 0 ) = 0.0;
rResult[0]( 0, 1 ) = 0.125 * ( 1.0 - rPoint[2] );
rResult[0]( 0, 2 ) = 0.125 * ( 1.0 - rPoint[1] );
rResult[0]( 1, 0 ) = 0.125 * ( 1.0 - rPoint[2] );
rResult[0]( 1, 1 ) = 0.0;
rResult[0]( 1, 2 ) = 0.125 * ( 1.0 - rPoint[0] );
rResult[0]( 2, 0 ) = 0.125 * ( 1.0 - rPoint[1] );
rResult[0]( 2, 1 ) = 0.125 * ( 1.0 - rPoint[0] );
rResult[0]( 2, 2 ) = 0.0;

rResult[1]( 0, 0 ) = 0.0;
rResult[1]( 0, 1 ) = -0.125 * ( 1.0 - rPoint[2] );
rResult[1]( 0, 2 ) = -0.125 * ( 1.0 - rPoint[1] );
rResult[1]( 1, 0 ) = -0.125 * ( 1.0 - rPoint[2] );
rResult[1]( 1, 1 ) = 0.0;
rResult[1]( 1, 2 ) = 0.125 * ( 1.0 + rPoint[0] );
rResult[1]( 2, 0 ) = -0.125 * ( 1.0 - rPoint[1] );
rResult[1]( 2, 1 ) = 0.125 * ( 1.0 + rPoint[0] );
rResult[1]( 2, 2 ) = 0.0;

rResult[2]( 0, 0 ) = 0.0;
rResult[2]( 0, 1 ) = 0.125 * ( 1.0 - rPoint[2] );
rResult[2]( 0, 2 ) = -0.125 * ( 1.0 + rPoint[1] );
rResult[2]( 1, 0 ) = 0.125 * ( 1.0 - rPoint[2] );
rResult[2]( 1, 1 ) = 0.0;
rResult[2]( 1, 2 ) = -0.125 * ( 1.0 + rPoint[0] );
rResult[2]( 2, 0 ) = -0.125 * ( 1.0 + rPoint[1] );
rResult[2]( 2, 1 ) = -0.125 * ( 1.0 + rPoint[0] );
rResult[2]( 2, 2 ) = 0.0;

rResult[3]( 0, 0 ) = 0.0;
rResult[3]( 0, 1 ) = -0.125 * ( 1.0 - rPoint[2] );
rResult[3]( 0, 2 ) = 0.125 * ( 1.0 + rPoint[1] );
rResult[3]( 1, 0 ) = -0.125 * ( 1.0 - rPoint[2] );
rResult[3]( 1, 1 ) = 0.0;
rResult[3]( 1, 2 ) = -0.125 * ( 1.0 - rPoint[0] );
rResult[3]( 2, 0 ) = 0.125 * ( 1.0 + rPoint[1] );
rResult[3]( 2, 1 ) = -0.125 * ( 1.0 - rPoint[0] );
rResult[3]( 2, 2 ) = 0.0;

rResult[4]( 0, 0 ) = 0.0;
rResult[4]( 0, 1 ) = 0.125 * ( 1.0 + rPoint[2] );
rResult[4]( 0, 2 ) = -0.125 * ( 1.0 - rPoint[1] );
rResult[4]( 1, 0 ) = 0.125 * ( 1.0 + rPoint[2] );
rResult[4]( 1, 1 ) = 0.0;
rResult[4]( 1, 2 ) = -0.125 * ( 1.0 - rPoint[0] );
rResult[4]( 2, 0 ) = -0.125 * ( 1.0 - rPoint[1] );
rResult[4]( 2, 1 ) = -0.125 * ( 1.0 - rPoint[0] );
rResult[4]( 2, 2 ) = 0.0;

rResult[5]( 0, 0 ) = 0.0;
rResult[5]( 0, 1 ) = -0.125 * ( 1.0 + rPoint[2] );
rResult[5]( 0, 2 ) = 0.125 * ( 1.0 - rPoint[1] );
rResult[5]( 1, 0 ) = -0.125 * ( 1.0 + rPoint[2] );
rResult[5]( 1, 1 ) = 0.0;
rResult[5]( 1, 2 ) = -0.125 * ( 1.0 + rPoint[0] );
rResult[5]( 2, 0 ) = 0.125 * ( 1.0 - rPoint[1] );
rResult[5]( 2, 1 ) = -0.125 * ( 1.0 + rPoint[0] );
rResult[5]( 2, 2 ) = 0.0;

rResult[6]( 0, 0 ) = 0.0;
rResult[6]( 0, 1 ) = 0.125 * ( 1.0 + rPoint[2] );
rResult[6]( 0, 2 ) = 0.125 * ( 1.0 + rPoint[1] );
rResult[6]( 1, 0 ) = 0.125 * ( 1.0 + rPoint[2] );
rResult[6]( 1, 1 ) = 0.0;
rResult[6]( 1, 2 ) = 0.125 * ( 1.0 + rPoint[0] );
rResult[6]( 2, 0 ) = 0.125 * ( 1.0 + rPoint[1] );
rResult[6]( 2, 1 ) = 0.125 * ( 1.0 + rPoint[0] );
rResult[6]( 2, 2 ) = 0.0;

rResult[7]( 0, 0 ) = 0.0;
rResult[7]( 0, 1 ) = -0.125 * ( 1.0 + rPoint[2] );
rResult[7]( 0, 2 ) = -0.125 * ( 1.0 + rPoint[1] );
rResult[7]( 1, 0 ) = -0.125 * ( 1.0 + rPoint[2] );
rResult[7]( 1, 1 ) = 0.0;
rResult[7]( 1, 2 ) = 0.125 * ( 1.0 - rPoint[0] );
rResult[7]( 2, 0 ) = -0.125 * ( 1.0 + rPoint[1] );
rResult[7]( 2, 1 ) = 0.125 * ( 1.0 - rPoint[0] );
rResult[7]( 2, 2 ) = 0.0;

return rResult;
}




std::string Info() const override
{
return "3 dimensional hexahedra with eight nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "3 dimensional hexahedra with eight nodes in 3D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
rOStream << "    Jacobian in the origin\t : " << jacobian;
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

Hexahedra3D8(): BaseType( PointsArrayType(), &msGeometryData ) {}





static Matrix ShapeFunctionsLocalGradients( const CoordinatesArrayType& rPoint )
{
Matrix result = ZeroMatrix( 8, 3 );
result( 0, 0 ) = -0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 - rPoint[2] );
result( 0, 1 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[2] );
result( 0, 2 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[1] );
result( 1, 0 ) =  0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 - rPoint[2] );
result( 1, 1 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[2] );
result( 1, 2 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[1] );
result( 2, 0 ) =  0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 - rPoint[2] );
result( 2, 1 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[2] );
result( 2, 2 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] );
result( 3, 0 ) = -0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 - rPoint[2] );
result( 3, 1 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[2] );
result( 3, 2 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[1] );
result( 4, 0 ) = -0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 + rPoint[2] );
result( 4, 1 ) = -0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[2] );
result( 4, 2 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 - rPoint[1] );
result( 5, 0 ) =  0.125 * ( 1.0 - rPoint[1] ) * ( 1.0 + rPoint[2] );
result( 5, 1 ) = -0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] );
result( 5, 2 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 - rPoint[1] );
result( 6, 0 ) =  0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] );
result( 6, 1 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] );
result( 6, 2 ) =  0.125 * ( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] );
result( 7, 0 ) = -0.125 * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] );
result( 7, 1 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[2] );
result( 7, 2 ) =  0.125 * ( 1.0 - rPoint[0] ) * ( 1.0 + rPoint[1] );
return result;
}




static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType& integration_points = all_integration_points[static_cast<int>(ThisMethod)];

const int integration_points_number = integration_points.size();
const int points_number = 8;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
shape_function_values( pnt, 0 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
shape_function_values( pnt, 1 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
shape_function_values( pnt, 2 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
shape_function_values( pnt, 3 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
shape_function_values( pnt, 4 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
shape_function_values( pnt, 5 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
shape_function_values( pnt, 6 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
shape_function_values( pnt, 7 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
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
Matrix& result = d_shape_f_values[pnt];
result = ZeroMatrix( 8, 3 );
result( 0, 0 ) =
-0.125 * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
result( 0, 1 ) =
-0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Z() );
result( 0, 2 ) =
-0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
result( 1, 0 ) =
0.125 * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
result( 1, 1 ) =
-0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Z() );
result( 1, 2 ) =
-0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
result( 2, 0 ) =
0.125 * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
result( 2, 1 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Z() );
result( 2, 2 ) =
-0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
result( 3, 0 ) =
-0.125 * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() );
result( 3, 1 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Z() );
result( 3, 2 ) =
-0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
result( 4, 0 ) =
-0.125 * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
result( 4, 1 ) =
-0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() );
result( 4, 2 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
result( 5, 0 ) =
0.125 * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
result( 5, 1 ) =
-0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() );
result( 5, 2 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
result( 6, 0 ) =
0.125 * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
result( 6, 1 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() );
result( 6, 2 ) =
0.125 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
result( 7, 0 ) =
-0.125 * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() );
result( 7, 1 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() );
result( 7, 2 ) =
0.125 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
}

return d_shape_f_values;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < HexahedronGaussLegendreIntegrationPoints1, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints2, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints3, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints4, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints5, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLobattoIntegrationPoints1, 3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLobattoIntegrationPoints2, 3, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 )
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
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Hexahedra3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 )
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Hexahedra3D8;




};





template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream, Hexahedra3D8<TPointType>& rThis );


template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream, const Hexahedra3D8<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}


template<class TPointType> const
GeometryData Hexahedra3D8<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Hexahedra3D8<TPointType>::AllIntegrationPoints(),
Hexahedra3D8<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Hexahedra3D8<TPointType>::msGeometryDimension(3, 3);

}
