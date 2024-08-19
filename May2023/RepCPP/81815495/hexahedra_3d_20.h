
#pragma once



#include "geometries/quadrilateral_3d_8.h"
#include "utilities/integration_utilities.h"
#include "integration/hexahedron_gauss_legendre_integration_points.h"

namespace Kratos
{

template<class TPointType> class Hexahedra3D20 : public Geometry<TPointType>
{
public:




typedef Geometry<TPointType> BaseType;


typedef Line3D3<TPointType> EdgeType;
typedef Quadrilateral3D8<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Hexahedra3D20 );


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


typedef typename BaseType::NormalType NormalType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;


typedef Matrix MatrixType;



Hexahedra3D20( const PointType& Point1, const PointType& Point2,
const PointType& Point3, const PointType& Point4,
const PointType& Point5, const PointType& Point6,
const PointType& Point7, const PointType& Point8,
const PointType& Point9, const PointType& Point10,
const PointType& Point11, const PointType& Point12,
const PointType& Point13, const PointType& Point14,
const PointType& Point15, const PointType& Point16,
const PointType& Point17, const PointType& Point18,
const PointType& Point19, const PointType& Point20
)
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
this->Points().push_back( typename PointType::Pointer( new PointType( Point9 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point10 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point11 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point12 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point13 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point14 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point15 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point16 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point17 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point18 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point19 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point20 ) ) );
}

Hexahedra3D20( typename PointType::Pointer pPoint1,
typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3,
typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5,
typename PointType::Pointer pPoint6,
typename PointType::Pointer pPoint7,
typename PointType::Pointer pPoint8,
typename PointType::Pointer pPoint9,
typename PointType::Pointer pPoint10,
typename PointType::Pointer pPoint11,
typename PointType::Pointer pPoint12,
typename PointType::Pointer pPoint13,
typename PointType::Pointer pPoint14,
typename PointType::Pointer pPoint15,
typename PointType::Pointer pPoint16,
typename PointType::Pointer pPoint17,
typename PointType::Pointer pPoint18,
typename PointType::Pointer pPoint19,
typename PointType::Pointer pPoint20 )
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
this->Points().push_back( pPoint9 );
this->Points().push_back( pPoint10 );
this->Points().push_back( pPoint11 );
this->Points().push_back( pPoint12 );
this->Points().push_back( pPoint13 );
this->Points().push_back( pPoint14 );
this->Points().push_back( pPoint15 );
this->Points().push_back( pPoint16 );
this->Points().push_back( pPoint17 );
this->Points().push_back( pPoint18 );
this->Points().push_back( pPoint19 );
this->Points().push_back( pPoint20 );
}

Hexahedra3D20( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF(this->PointsNumber() != 20) << "Invalid points number. Expected 20, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D20(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 20 ) << "Invalid points number. Expected 20, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D20(
const std::string& GeometryName,
const PointsArrayType& rThisPoints
) : BaseType( GeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 20) << "Invalid points number. Expected 20, given " << this->PointsNumber() << std::endl;
}


Hexahedra3D20( Hexahedra3D20 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Hexahedra3D20( Hexahedra3D20<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Hexahedra3D20() override {}


GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Hexahedra;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Hexahedra3D20;
}




Hexahedra3D20& operator=( const Hexahedra3D20& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Hexahedra3D20& operator=( Hexahedra3D20<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Hexahedra3D20( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Hexahedra3D20( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}




double Length() const override
{
return sqrt( fabs( this->DeterminantOfJacobian( PointType() ) ) );
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


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) )
{
if ( std::abs( rResult[1] ) <= (1.0 + Tolerance) )
{
if ( std::abs( rResult[2] ) <= (1.0 + Tolerance) )
{
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
this->pGetPoint( 1 ),
this->pGetPoint( 8 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 9 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 10 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 11 ) ) ) );

edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ),
this->pGetPoint( 16 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 5 ),
this->pGetPoint( 6 ),
this->pGetPoint( 17 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 6 ),
this->pGetPoint( 7 ),
this->pGetPoint( 18 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 7 ),
this->pGetPoint( 4 ),
this->pGetPoint( 19 ) ) ) );

edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 4 ),
this->pGetPoint( 12 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 5 ),
this->pGetPoint( 13 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ),
this->pGetPoint( 14 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 7 ),
this->pGetPoint( 15 ) ) ) );
return edges;
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
this->pGetPoint( 0 ),
this->pGetPoint( 10 ),
this->pGetPoint( 9 ),
this->pGetPoint( 8 ),
this->pGetPoint( 11 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 5 ),
this->pGetPoint( 4 ),
this->pGetPoint( 8 ),
this->pGetPoint( 13 ),
this->pGetPoint( 16 ),
this->pGetPoint( 12 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ),
this->pGetPoint( 5 ),
this->pGetPoint( 1 ),
this->pGetPoint( 14 ),
this->pGetPoint( 17 ),
this->pGetPoint( 13 ),
this->pGetPoint( 9 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 6 ),
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 14 ),
this->pGetPoint( 18 ),
this->pGetPoint( 10 ),
this->pGetPoint( 15 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 4 ),
this->pGetPoint( 15 ),
this->pGetPoint( 11 ),
this->pGetPoint( 12 ),
this->pGetPoint( 19 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ),
this->pGetPoint( 6 ),
this->pGetPoint( 7 ),
this->pGetPoint( 16 ),
this->pGetPoint( 17 ),
this->pGetPoint( 18 ),
this->pGetPoint( 19 ) ) ) );
return faces;
}




double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0 :
return -(( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1] )*( 2.0
- rPoint[0] + rPoint[1] - rPoint[2] )*( 1.0 + rPoint[2] ) ) / 8.0;
case 1 :
return -(( 1.0 + rPoint[0] )
*( 1.0 + rPoint[1] )*( 2.0
- rPoint[0] - rPoint[1] - rPoint[2] )*( 1.0 + rPoint[2] ) ) / 8.0;
case 2 :
return -(( 1.0 + rPoint[0] )
*( 1.0 + rPoint[1] )*( 1.0 - rPoint[2] )*( 2.0
- rPoint[0] - rPoint[1] + rPoint[2] ) ) / 8.0;
case 3:
return -(( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1] )*( 1.0 - rPoint[2] )*( 2.0
- rPoint[0] + rPoint[1] + rPoint[2] ) ) / 8.0;
case 4 :
return -(( 1.0 - rPoint[0] )
*( 1.0 - rPoint[1] )*( 2.0
+ rPoint[0] + rPoint[1] - rPoint[2] )*( 1.0 + rPoint[2] ) ) / 8.0;
case 5 :
return -(( 1.0 - rPoint[0] )
*( 1.0 + rPoint[1] )*( 2.0
+ rPoint[0] - rPoint[1] - rPoint[2] )*( 1.0 + rPoint[2] ) ) / 8.0;
case 6 :
return -(( 1.0 - rPoint[0] )*( 1.0
+ rPoint[1] )*( 1.0 - rPoint[2] )*( 2.0
+ rPoint[0] - rPoint[1] + rPoint[2] ) ) / 8.0;
case 7 :
return -(( 1.0 - rPoint[0] )*( 1.0 - rPoint[1] )
*( 1.0 - rPoint[2] )*( 2.0 + rPoint[0]
+ rPoint[1] + rPoint[2] ) ) / 8.0;
case 8 :
return (( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] )*( 1.0 + rPoint[2] ) ) / 4.0 ;
case 9 :
return (( 1.0 + rPoint[0] )*( 1.0 + rPoint[1] )
*( 1.0 - rPoint[2]*rPoint[2] ) ) / 4.0 ;
case 10 :
return (( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] )*( 1.0 - rPoint[2] ) ) / 4.0 ;
case 11 :
return (( 1.0 + rPoint[0] )*( 1.0 - rPoint[1] )
*( 1.0 - rPoint[2]*rPoint[2] ) ) / 4.0 ;
case 12 :
return (( 1.0 -rPoint[0]
*rPoint[0] )*( 1.0 - rPoint[1] )*( 1.0 + rPoint[2] ) ) / 4.0 ;
case 13 :
return (( 1.0 -rPoint[0]
*rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 + rPoint[2] ) ) / 4.0 ;
case 14 :
return (( 1.0 -rPoint[0]
*rPoint[0] )*( 1.0 + rPoint[1] )*( 1.0 - rPoint[2] ) ) / 4.0 ;
case 15 :
return (( 1.0 -rPoint[0]*rPoint[0] )
*( 1.0 - rPoint[1] )*( 1.0 - rPoint[2] ) ) / 4.0;
case 16 :
return (( 1.0 -rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] )*( 1.0 + rPoint[2] ) ) / 4.0 ;
case 17 :
return (( 1.0 -rPoint[0] )*( 1.0 + rPoint[1] )
*( 1.0 - rPoint[2]*rPoint[2] ) ) / 4.0 ;
case 18 :
return (( 1.0 -rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] )*( 1.0 - rPoint[2] ) ) / 4.0 ;
case 19 :
return (( 1.0 -rPoint[0] )
*( 1.0 - rPoint[1] )*( 1.0 - rPoint[2]*rPoint[2] ) ) / 4.0 ;

default:
KRATOS_ERROR << "Wrong index of shape function!" << *this  << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 20) rResult.resize(20,false);
rResult[0] = -(( 1.0 + rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 2.0
- rCoordinates[0] + rCoordinates[1] - rCoordinates[2] )*( 1.0 + rCoordinates[2] ) ) / 8.0;
rResult[1] = -(( 1.0 + rCoordinates[0] )
*( 1.0 + rCoordinates[1] )*( 2.0
- rCoordinates[0] - rCoordinates[1] - rCoordinates[2] )*( 1.0 + rCoordinates[2] ) ) / 8.0;
rResult[2] = -(( 1.0 + rCoordinates[0] )
*( 1.0 + rCoordinates[1] )*( 1.0 - rCoordinates[2] )*( 2.0
- rCoordinates[0] - rCoordinates[1] + rCoordinates[2] ) ) / 8.0;
rResult[3] = -(( 1.0 + rCoordinates[0] )
*( 1.0 - rCoordinates[1] )*( 1.0 - rCoordinates[2] )*( 2.0
- rCoordinates[0] + rCoordinates[1] + rCoordinates[2] ) ) / 8.0;
rResult[4] = -(( 1.0 - rCoordinates[0] )
*( 1.0 - rCoordinates[1] )*( 2.0
+ rCoordinates[0] + rCoordinates[1] - rCoordinates[2] )*( 1.0 + rCoordinates[2] ) ) / 8.0;
rResult[5] = -(( 1.0 - rCoordinates[0] )
*( 1.0 + rCoordinates[1] )*( 2.0
+ rCoordinates[0] - rCoordinates[1] - rCoordinates[2] )*( 1.0 + rCoordinates[2] ) ) / 8.0;
rResult[6] = -(( 1.0 - rCoordinates[0] )*( 1.0
+ rCoordinates[1] )*( 1.0 - rCoordinates[2] )*( 2.0
+ rCoordinates[0] - rCoordinates[1] + rCoordinates[2] ) ) / 8.0;
rResult[7] = -(( 1.0 - rCoordinates[0] )*( 1.0 - rCoordinates[1] )
*( 1.0 - rCoordinates[2] )*( 2.0 + rCoordinates[0]
+ rCoordinates[1] + rCoordinates[2] ) ) / 8.0;
rResult[8] = (( 1.0 + rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ) / 4.0 ;
rResult[9] = (( 1.0 + rCoordinates[0] )*( 1.0 + rCoordinates[1] )
*( 1.0 - rCoordinates[2]*rCoordinates[2] ) ) / 4.0 ;
rResult[10] = (( 1.0 + rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ) / 4.0 ;
rResult[11] = (( 1.0 + rCoordinates[0] )*( 1.0 - rCoordinates[1] )
*( 1.0 - rCoordinates[2]*rCoordinates[2] ) ) / 4.0 ;
rResult[12] = (( 1.0 -rCoordinates[0]
*rCoordinates[0] )*( 1.0 - rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ) / 4.0 ;
rResult[13] = (( 1.0 -rCoordinates[0]
*rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ) / 4.0 ;
rResult[14] = (( 1.0 -rCoordinates[0]
*rCoordinates[0] )*( 1.0 + rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ) / 4.0 ;
rResult[15] = (( 1.0 -rCoordinates[0]*rCoordinates[0] )
*( 1.0 - rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ) / 4.0;
rResult[16] = (( 1.0 -rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] )*( 1.0 + rCoordinates[2] ) ) / 4.0 ;
rResult[17] = (( 1.0 -rCoordinates[0] )*( 1.0 + rCoordinates[1] )
*( 1.0 - rCoordinates[2]*rCoordinates[2] ) ) / 4.0 ;
rResult[18] = (( 1.0 -rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] )*( 1.0 - rCoordinates[2] ) ) / 4.0 ;
rResult[19] = (( 1.0 -rCoordinates[0] )
*( 1.0 - rCoordinates[1] )*( 1.0 - rCoordinates[2]*rCoordinates[2] ) ) / 4.0 ;
return rResult;
}



std::string Info() const override
{
return "3 dimensional hexahedra with 20 nodes and quadratic shape functions in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "3 dimensional hexahedra with 20 nodes and quadratic shape functions in 3D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
rOStream << "    Jacobian in the origin\t : " << jacobian;
}



Matrix& ShapeFunctionsLocalGradients( Matrix& result,
const CoordinatesArrayType& rPoint ) const override
{
if ( result.size1() != 20 || result.size2() != 3 )
result.resize( 20, 3, false );

result( 0, 0 ) = (( -1.0 + rPoint[1] ) * ( 1.0 - 2.0 * rPoint[0] + rPoint[1] - rPoint[2] ) * ( 1.0
+ rPoint[2] ) ) / 8.0;

result( 0, 1 ) = -(( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] ) * ( -1.0 + rPoint[0] - 2.0
* rPoint[1] + rPoint[2] ) ) / 8.0;

result( 0, 2 ) = -(( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[0] - rPoint[1] + 2.0
* rPoint[2] ) ) / 8.0;

result( 1, 0 ) = (( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] ) * ( -1.0 + 2.0
* rPoint[0] + rPoint[1] + rPoint[2] ) ) / 8.0;

result( 1, 1 ) = (( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[2] ) * ( -1.0 + rPoint[0] + 2.0
* rPoint[1] + rPoint[2] ) ) / 8.0;

result( 1, 2 ) = (( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * ( -1.0 + rPoint[0] + rPoint[1] + 2.0
* rPoint[2] ) ) / 8.0;

result( 2, 0 ) = -(( 1.0 + rPoint[1] ) * ( -1.0 + 2.0 * rPoint[0] + rPoint[1] - rPoint[2] ) * ( -1.0
+ rPoint[2] ) ) / 8.0;

result( 2, 1 ) = -(( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[0] + 2.0 * rPoint[1] - rPoint[2] ) * ( -1.0
+ rPoint[2] ) ) / 8.0;

result( 2, 2 ) = -(( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * ( -1.0 + rPoint[0] + rPoint[1] - 2.0
* rPoint[2] ) ) / 8.0;

result( 3, 0 ) = -(( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] ) * ( 1.0 - 2.0
* rPoint[0] + rPoint[1] + rPoint[2] ) ) / 8.0;

result( 3, 1 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[0] - 2.0 * rPoint[1] - rPoint[2] ) * ( -1.0
+ rPoint[2] ) ) / 8.0;

result( 3, 2 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[0] - rPoint[1] - 2.0
* rPoint[2] ) ) / 8.0;

result( 4, 0 ) = -(( -1.0 + rPoint[1] ) * ( 1.0 + 2.0 * rPoint[0] + rPoint[1] - rPoint[2] ) * ( 1.0
+ rPoint[2] ) ) / 8.0;

result( 4, 1 ) = -(( -1.0 + rPoint[0] ) * ( 1.0 + rPoint[0] + 2.0 * rPoint[1] - rPoint[2] ) * ( 1.0
+ rPoint[2] ) ) / 8.0;

result( 4, 2 ) = -(( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * ( 1.0 + rPoint[0] + rPoint[1] - 2.0
* rPoint[2] ) ) / 8.0;

result( 5, 0 ) = -(( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] ) * ( -1.0 - 2.0
* rPoint[0] + rPoint[1] + rPoint[2] ) ) / 8.0;

result( 5, 1 ) = (( -1.0 + rPoint[0] ) * ( 1.0 + rPoint[0] - 2.0 * rPoint[1] - rPoint[2] ) * ( 1.0
+ rPoint[2] ) ) / 8.0;

result( 5, 2 ) = (( -1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[0] - rPoint[1] - 2.0
* rPoint[2] ) ) / 8.0;

result( 6, 0 ) = (( 1.0 + rPoint[1] ) * ( -1.0 - 2.0 * rPoint[0] + rPoint[1] - rPoint[2] ) * ( -1.0
+ rPoint[2] ) ) / 8.0;

result( 6, 1 ) = -(( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] ) * ( 1.0 + rPoint[0] - 2.0
* rPoint[1] + rPoint[2] ) ) / 8.0;

result( 6, 2 ) = -(( -1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[0] - rPoint[1] + 2.0
* rPoint[2] ) ) / 8.0;

result( 7, 0 ) = (( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] ) * ( 1.0 + 2.0
* rPoint[0] + rPoint[1] + rPoint[2] ) ) / 8.0;

result( 7, 1 ) = (( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] ) * ( 1.0 + rPoint[0] + 2.0
* rPoint[1] + rPoint[2] ) ) / 8.0;

result( 7, 2 ) = (( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * ( 1.0 + rPoint[0] + rPoint[1] + 2.0
* rPoint[2] ) ) / 8.0;

result( 8, 0 ) = -(( -1.0 + rPoint[1] * rPoint[1] ) * ( 1.0 + rPoint[2] ) ) / 4.0;

result( 8, 1 ) = -(( 1.0 + rPoint[0] ) * rPoint[1] * ( 1.0 + rPoint[2] ) ) / 2.0;

result( 8, 2 ) = -(( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] * rPoint[1] ) ) / 4.0;

result( 9, 0 ) = -(( 1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 9, 1 ) = -(( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 9, 2 ) = -(( 1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * rPoint[2] ) / 2.0;

result( 10, 0 ) = (( -1.0 + rPoint[1] * rPoint[1] ) * ( -1.0 + rPoint[2] ) ) / 4.0;

result( 10, 1 ) = (( 1.0 + rPoint[0] ) * rPoint[1] * ( -1.0 + rPoint[2] ) ) / 2.0;

result( 10, 2 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] * rPoint[1] ) ) / 4.0;

result( 11, 0 ) = (( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 11, 1 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 11, 2 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * rPoint[2] ) / 2.0;

result( 12, 0 ) = ( rPoint[0] * ( -1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] ) ) / 2.0;

result( 12, 1 ) = (( -1.0 + rPoint[0] * rPoint[0] ) * ( 1.0 + rPoint[2] ) ) / 4.0;

result( 12, 2 ) = (( -1.0 + rPoint[0] * rPoint[0] ) * ( -1.0 + rPoint[1] ) ) / 4.0;

result( 13, 0 ) = -( rPoint[0] * ( 1.0 + rPoint[1] ) * ( 1.0 + rPoint[2] ) ) / 2.0;

result( 13, 1 ) = -(( -1.0 + rPoint[0] * rPoint[0] ) * ( 1.0 + rPoint[2] ) ) / 4.0;

result( 13, 2 ) = -(( -1.0 + rPoint[0] * rPoint[0] ) * ( 1.0 + rPoint[1] ) ) / 4.0;

result( 14, 0 ) = ( rPoint[0] * ( 1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] ) ) / 2.0;

result( 14, 1 ) = (( -1.0 + rPoint[0] * rPoint[0] ) * ( -1.0 + rPoint[2] ) ) / 4.0;

result( 14, 2 ) = (( -1.0 + rPoint[0] * rPoint[0] ) * ( 1.0 + rPoint[1] ) ) / 4.0;

result( 15, 0 ) = -( rPoint[0] * ( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] ) ) / 2.0;

result( 15, 1 ) = -(( -1.0 + rPoint[0] * rPoint[0] ) * ( -1.0 + rPoint[2] ) ) / 4.0;

result( 15, 2 ) = -(( -1.0 + rPoint[0] * rPoint[0] ) * ( -1.0 + rPoint[1] ) ) / 4.0;

result( 16, 0 ) = (( -1.0 + rPoint[1] * rPoint[1] ) * ( 1.0 + rPoint[2] ) ) / 4.0;

result( 16, 1 ) = (( -1.0 + rPoint[0] ) * rPoint[1] * ( 1.0 + rPoint[2] ) ) / 2.0;

result( 16, 2 ) = (( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] * rPoint[1] ) ) / 4.0;

result( 17, 0 ) = (( 1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 17, 1 ) = (( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 17, 2 ) = (( -1.0 + rPoint[0] ) * ( 1.0 + rPoint[1] ) * rPoint[2] ) / 2.0;

result( 18, 0 ) = -(( -1.0 + rPoint[1] * rPoint[1] ) * ( -1.0 + rPoint[2] ) ) / 4.0;

result( 18, 1 ) = -(( -1.0 + rPoint[0] ) * rPoint[1] * ( -1.0 + rPoint[2] ) ) / 2.0;

result( 18, 2 ) = -(( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] * rPoint[1] ) ) / 4.0;

result( 19, 0 ) = -(( -1.0 + rPoint[1] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 19, 1 ) = -(( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[2] * rPoint[2] ) ) / 4.0;

result( 19, 2 ) = -(( -1.0 + rPoint[0] ) * ( -1.0 + rPoint[1] ) * rPoint[2] ) / 2.0;

return( result );
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

Hexahedra3D20(): BaseType( PointsArrayType(), &msGeometryData ) {}






static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType  all_integration_points =
AllIntegrationPoints();
const IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 20;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
shape_function_values(pnt, 0 ) =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 2.0
- integration_points[pnt].X() + integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0 + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 1 ) =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 2.0
- integration_points[pnt].X() - integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0 + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 2 ) =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 1.0 - integration_points[pnt].Z() ) * ( 2.0
- integration_points[pnt].X() - integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 3 ) =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 1.0 - integration_points[pnt].Z() )
* ( 2.0 - integration_points[pnt].X()
+ integration_points[pnt].Y() + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 4 ) =
-(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 2.0
+ integration_points[pnt].X() + integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0 + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 5 ) =
-(( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 2.0
+ integration_points[pnt].X() - integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0 + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 6 ) =
-(( 1.0 - integration_points[pnt].X() ) * ( 1.0
+ integration_points[pnt].Y() ) * ( 1.0 - integration_points[pnt].Z() ) * ( 2.0
+ integration_points[pnt].X() - integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 7 ) =
-(( 1.0 - integration_points[pnt].X() ) * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() ) * ( 2.0 + integration_points[pnt].X()
+ integration_points[pnt].Y() + integration_points[pnt].Z() ) ) / 8.0;
shape_function_values(pnt, 8 ) =
(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() * integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 9 ) =
(( 1.0 + integration_points[pnt].X() ) * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() * integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 10 ) =
(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() * integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 11 ) =
(( 1.0 + integration_points[pnt].X() ) * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() * integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 12 ) =
(( 1.0 - integration_points[pnt].X()
* integration_points[pnt].X() ) * ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 13 ) =
(( 1.0 - integration_points[pnt].X()
* integration_points[pnt].X() ) * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 14 ) =
(( 1.0 - integration_points[pnt].X()
* integration_points[pnt].X() ) * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 15 ) =
(( 1.0 - integration_points[pnt].X() * integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 1.0 - integration_points[pnt].Z() ) ) / 4.0;
shape_function_values(pnt, 16 ) =
(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() * integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 17 ) =
(( 1.0 - integration_points[pnt].X() ) * ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() * integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 18 ) =
(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() * integration_points[pnt].Y() )
* ( 1.0 - integration_points[pnt].Z() ) ) / 4.0 ;
shape_function_values(pnt, 19 ) =
(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 1.0
- integration_points[pnt].Z() * integration_points[pnt].Z() ) ) / 4.0 ;
}

return shape_function_values;
}



static ShapeFunctionsGradientsType
CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
const IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Matrix result = ZeroMatrix( 20, 3 );

result( 0, 0 ) = (( -1.0 + integration_points[pnt].Y() )
* ( 1.0 - 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 0, 1 ) = -(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() ) * ( -1.0
+ integration_points[pnt].X() - 2.0
* integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 0, 2 ) = -(( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() ) * ( -1.0
+ integration_points[pnt].X()
- integration_points[pnt].Y() + 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 1, 0 ) = (( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() )
* ( -1.0 + 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 1, 1 ) = (( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() ) * ( -1.0
+ integration_points[pnt].X() + 2.0
* integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 1, 2 ) = (( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( -1.0
+ integration_points[pnt].X()
+ integration_points[pnt].Y() + 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 2, 0 ) = -(( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( -1.0 + integration_points[pnt].Z() ) ) / 8.0;
result( 2, 1 ) = -(( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].X() + 2.0
* integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( -1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 2, 2 ) = -(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].X()
+ integration_points[pnt].Y() - 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 3, 0 ) = -(( -1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z() ) * ( 1.0 - 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 3, 1 ) = (( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].X() - 2.0
* integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( -1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 3, 2 ) = (( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].X()
- integration_points[pnt].Y() - 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 4, 0 ) = -(( -1.0 + integration_points[pnt].Y() )
* ( 1.0 + 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0 + integration_points[pnt].Z() ) ) / 8.0;
result( 4, 1 ) = -(( -1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].X() + 2.0
* integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 4, 2 ) = -(( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() ) * ( 1.0
+ integration_points[pnt].X()
+ integration_points[pnt].Y() - 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 5, 0 ) = -(( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) * ( -1.0 - 2.0
* integration_points[pnt].X()
+ integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 5, 1 ) = (( -1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].X() - 2.0
* integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( 1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 5, 2 ) = (( -1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 1.0
+ integration_points[pnt].X()
- integration_points[pnt].Y() - 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 6, 0 ) = (( 1.0 + integration_points[pnt].Y() )
* ( -1.0 - 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y()
- integration_points[pnt].Z() ) * ( -1.0
+ integration_points[pnt].Z() ) ) / 8.0;
result( 6, 1 ) = -(( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z() )
* ( 1.0 + integration_points[pnt].X()
- 2.0 * integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 6, 2 ) = -(( -1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 1.0
+ integration_points[pnt].X()
- integration_points[pnt].Y() + 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 7, 0 ) = (( -1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z() ) * ( 1.0 + 2.0
* integration_points[pnt].X() + integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 7, 1 ) = (( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z() )
* ( 1.0 + integration_points[pnt].X() + 2.0 * integration_points[pnt].Y()
+ integration_points[pnt].Z() ) ) / 8.0;
result( 7, 2 ) = (( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() ) * ( 1.0 + integration_points[pnt].X()
+ integration_points[pnt].Y() + 2.0
* integration_points[pnt].Z() ) ) / 8.0;

result( 8, 0 ) = -(( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 8, 1 ) = -(( 1.0 + integration_points[pnt].X() )
* integration_points[pnt].Y()
* ( 1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 8, 2 ) = -(( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 4.0;

result( 9, 0 ) = -(( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 9, 1 ) = -(( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 9, 2 ) = -(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* integration_points[pnt].Z() ) / 2.0;

result( 10, 0 ) = (( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 10, 1 ) = (( 1.0 + integration_points[pnt].X() )
* integration_points[pnt].Y()
* ( -1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 10, 2 ) = (( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 4.0;

result( 11, 0 ) = (( -1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 11, 1 ) = (( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 11, 2 ) = (( 1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() )
* integration_points[pnt].Z() ) / 2.0;

result( 12, 0 ) = ( integration_points[pnt].X()
* ( -1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 12, 1 ) = (( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 12, 2 ) = (( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() ) ) / 4.0;

result( 13, 0 ) = -( integration_points[pnt].X()
* ( 1.0 + integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 13, 1 ) = -(( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 13, 2 ) = -(( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) ) / 4.0;

result( 14, 0 ) = ( integration_points[pnt].X()
* ( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 14, 1 ) = (( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 14, 2 ) = (( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) ) / 4.0;

result( 15, 0 ) = -( integration_points[pnt].X() * ( -1.0
+ integration_points[pnt].Y() ) * ( -1.0
+ integration_points[pnt].Z() ) ) / 2.0;
result( 15, 1 ) = -(( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() ) * ( -1.0
+ integration_points[pnt].Z() ) ) / 4.0;
result( 15, 2 ) = -(( -1.0 + integration_points[pnt].X()
* integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() ) ) / 4.0;

result( 16, 0 ) = (( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 16, 1 ) = (( -1.0 + integration_points[pnt].X() )
* integration_points[pnt].Y()
* ( 1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 16, 2 ) = (( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 4.0;

result( 17, 0 ) = (( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 17, 1 ) = (( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 17, 2 ) = (( -1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() )
* integration_points[pnt].Z() ) / 2.0;

result( 18, 0 ) = -(( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z() ) ) / 4.0;
result( 18, 1 ) = -(( -1.0 + integration_points[pnt].X() )
* integration_points[pnt].Y()
* ( -1.0 + integration_points[pnt].Z() ) ) / 2.0;
result( 18, 2 ) = -(( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 4.0;

result( 19, 0 ) = -(( -1.0 + integration_points[pnt].Y() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 19, 1 ) = -(( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Z()
* integration_points[pnt].Z() ) ) / 4.0;
result( 19, 2 ) = -(( -1.0 + integration_points[pnt].X() )
* ( -1.0 + integration_points[pnt].Y() )
* integration_points[pnt].Z() ) / 2.0;

d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}


static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < HexahedronGaussLegendreIntegrationPoints1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints5,
3, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}


static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_5 )
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
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D20<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Hexahedra3D20;




};





template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream, Hexahedra3D20<TPointType>& rThis );


template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream, const Hexahedra3D20<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}

template<class TPointType> const
GeometryData Hexahedra3D20<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_3,
Hexahedra3D20<TPointType>::AllIntegrationPoints(),
Hexahedra3D20<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Hexahedra3D20<TPointType>::msGeometryDimension(3, 3);

}
