
#pragma once



#include "geometries/line_3d_3.h"
#include "utilities/integration_utilities.h"
#include "integration/quadrilateral_gauss_legendre_integration_points.h"

namespace Kratos
{

template<class TPointType> class Quadrilateral3D8
: public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;


typedef Line3D3<TPointType> EdgeType;


KRATOS_CLASS_POINTER_DEFINITION( Quadrilateral3D8 );


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef typename BaseType::GeometriesArrayType GeometriesArrayType;


typedef TPointType PointType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;



typedef typename BaseType::IndexType IndexType;


typedef typename BaseType::SizeType SizeType;


typedef  typename BaseType::PointsArrayType PointsArrayType;


typedef typename BaseType::IntegrationPointType IntegrationPointType;


typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


typedef typename BaseType::IntegrationPointsContainerType IntegrationPointsContainerType;


typedef typename BaseType::ShapeFunctionsValuesContainerType
ShapeFunctionsValuesContainerType;


typedef typename BaseType::ShapeFunctionsLocalGradientsContainerType
ShapeFunctionsLocalGradientsContainerType;


typedef typename BaseType::JacobiansType JacobiansType;


typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef typename BaseType::ShapeFunctionsSecondDerivativesType
ShapeFunctionsSecondDerivativesType;



typedef typename BaseType::ShapeFunctionsThirdDerivativesType
ShapeFunctionsThirdDerivativesType;


typedef typename BaseType::NormalType NormalType;



Quadrilateral3D8( typename PointType::Pointer pPoint1, typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3, typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5, typename PointType::Pointer pPoint6,
typename PointType::Pointer pPoint7, typename PointType::Pointer pPoint8 )
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

Quadrilateral3D8( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 8 ) << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral3D8(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 8 ) << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral3D8(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 8) << "Invalid points number. Expected 8, given " << this->PointsNumber() << std::endl;
}


Quadrilateral3D8( Quadrilateral3D8 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Quadrilateral3D8( Quadrilateral3D8<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Quadrilateral3D8() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Quadrilateral;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Quadrilateral3D8;
}




Quadrilateral3D8& operator=( const Quadrilateral3D8& rOther )
{
BaseType::operator=( rOther );

return *this;
}


template<class TOtherPointType>
Quadrilateral3D8& operator=( Quadrilateral3D8<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Quadrilateral3D8( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Quadrilateral3D8( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}



SizeType PointsNumberInDirection(IndexType LocalDirectionIndex) const override
{
if ((LocalDirectionIndex == 0) || (LocalDirectionIndex == 1)) {
return 3;
}
KRATOS_ERROR << "Possible direction index reaches from 0-1. Given direction index: "
<< LocalDirectionIndex << std::endl;
}



double Length() const override
{
return std::sqrt( std::abs( this->DeterminantOfJacobian( PointType() ) ) );
}


double Area() const override
{
const IntegrationMethod integration_method = msGeometryData.DefaultIntegrationMethod();
return IntegrationUtilities::ComputeDomainSize(*this, integration_method);
}


double Volume() const override
{
KRATOS_WARNING("Quadrilateral3D8") << "Method not well defined. Replace with DomainSize() instead. This method preserves current behaviour but will be changed in June 2023 (returning error instead)" << std::endl;
return Area();
}


double DomainSize() const override
{
return Area();
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
PointLocalCoordinates( rResult, rPoint );

if ( (rResult[0] >= (-1.0-Tolerance)) && (rResult[0] <= (1.0+Tolerance)) )
{
if ( (rResult[1] >= (-1.0-Tolerance)) && (rResult[1] <= (1.0+Tolerance)) )
{
return true;
}
}

return false;
}


CoordinatesArrayType& PointLocalCoordinates(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint
) const override
{
const double tol = 1.0e-8;
const int maxiter = 1000;

std::vector< unsigned int> orientation( 3 );

double dummy = this->GetPoint( 0 ).X();

if ( std::abs( this->GetPoint( 1 ).X() - dummy ) <= tol && std::abs( this->GetPoint( 2 ).X() - dummy ) <= tol && std::abs( this->GetPoint( 3 ).X() - dummy ) <= tol )
orientation[0] = 0;

dummy = this->GetPoint( 0 ).Y();

if ( std::abs( this->GetPoint( 1 ).Y() - dummy ) <= tol && std::abs( this->GetPoint( 2 ).Y() - dummy ) <= tol && std::abs( this->GetPoint( 3 ).Y() - dummy ) <= tol )
orientation[0] = 1;

dummy = this->GetPoint( 0 ).Z();

if ( std::abs( this->GetPoint( 1 ).Z() - dummy ) <= tol && std::abs( this->GetPoint( 2 ).Z() - dummy ) <= tol && std::abs( this->GetPoint( 3 ).Z() - dummy ) <= tol )
orientation[0] = 2;

switch ( orientation[0] )
{
case 0:
orientation[0] = 1;
orientation[1] = 2;
orientation[2] = 0;
break;
case 1:
orientation[0] = 0;
orientation[1] = 2;
orientation[2] = 1;
break;
case 2:
orientation[0] = 0;
orientation[1] = 1;
orientation[2] = 2;
break;
default:
orientation[0] = 0;
orientation[1] = 1;
orientation[2] = 2;
}

Matrix J = ZeroMatrix( 2, 2 );

Matrix invJ = ZeroMatrix( 2, 2 );

if ( rResult.size() != 2 )
rResult.resize( 2, false );

rResult = ZeroVector( 2 );

Vector DeltaXi = ZeroVector( 2 );

CoordinatesArrayType CurrentGlobalCoords( ZeroVector( 3 ) );

for ( int k = 0; k < maxiter; k++ )
{
CurrentGlobalCoords = ZeroVector( 3 );
this->GlobalCoordinates( CurrentGlobalCoords, rResult );
noalias( CurrentGlobalCoords ) = rPoint - CurrentGlobalCoords;

noalias(J) = ZeroMatrix(2, 2);

Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(
shape_functions_gradients, rResult );

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
Point dummyPoint = this->GetPoint( i );
J( 0, 0 ) += ( dummyPoint[orientation[0]] ) * ( shape_functions_gradients( i, 0 ) );
J( 0, 1 ) += ( dummyPoint[orientation[0]] ) * ( shape_functions_gradients( i, 1 ) );
J( 1, 0 ) += ( dummyPoint[orientation[1]] ) * ( shape_functions_gradients( i, 0 ) );
J( 1, 1 ) += ( dummyPoint[orientation[1]] ) * ( shape_functions_gradients( i, 1 ) );
}

double det_j = J( 0, 0 ) * J( 1, 1 ) - J( 0, 1 ) * J( 1, 0 );

invJ( 0, 0 ) = ( J( 1, 1 ) ) / ( det_j );

invJ( 1, 0 ) = -( J( 1, 0 ) ) / ( det_j );

invJ( 0, 1 ) = -( J( 0, 1 ) ) / ( det_j );

invJ( 1, 1 ) = ( J( 0, 0 ) ) / ( det_j );


DeltaXi( 0 ) = invJ( 0, 0 ) * CurrentGlobalCoords( orientation[0] ) + invJ( 0, 1 ) * CurrentGlobalCoords( orientation[1] );

DeltaXi( 1 ) = invJ( 1, 0 ) * CurrentGlobalCoords( orientation[0] ) + invJ( 1, 1 ) * CurrentGlobalCoords( orientation[1] );

noalias( rResult ) += DeltaXi;

if ( MathUtils<double>::Norm3( DeltaXi ) > 30 )
{
break;
}

if ( MathUtils<double>::Norm3( DeltaXi ) < tol )
{
if ( !( std::abs( CurrentGlobalCoords( orientation[2] ) ) <= tol ) )
rResult( 0 ) = 2.0;

break;
}
}

return( rResult );
}




JacobiansType& Jacobian( JacobiansType& rResult,
IntegrationMethod ThisMethod ) const override
{
ShapeFunctionsGradientsType shape_functions_gradients =
CalculateShapeFunctionsIntegrationPointsLocalGradients( ThisMethod );
Matrix shape_functions_values =
CalculateShapeFunctionsIntegrationPointsValues( ThisMethod );

if ( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( this->IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ )
{
Matrix jacobian = ZeroMatrix( 3, 2 );

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
jacobian( 0, 0 ) += ( this->GetPoint( i ).X() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 0, 1 ) += ( this->GetPoint( i ).X() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 1, 0 ) += ( this->GetPoint( i ).Y() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 1, 1 ) += ( this->GetPoint( i ).Y() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 2, 0 ) += ( this->GetPoint( i ).Z() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 2, 1 ) += ( this->GetPoint( i ).Z() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
}

rResult[pnt] = jacobian;
}

return rResult;
}



JacobiansType& Jacobian( JacobiansType& rResult,
IntegrationMethod ThisMethod,
Matrix & DeltaPosition ) const override
{
ShapeFunctionsGradientsType shape_functions_gradients =
CalculateShapeFunctionsIntegrationPointsLocalGradients( ThisMethod );
Matrix shape_functions_values =
CalculateShapeFunctionsIntegrationPointsValues( ThisMethod );

if ( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( this->IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ )
{
Matrix jacobian = ZeroMatrix( 3, 2 );

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
jacobian( 0, 0 ) += ( this->GetPoint( i ).X() - DeltaPosition(i,0)  ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 0, 1 ) += ( this->GetPoint( i ).X() - DeltaPosition(i,0)  ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 1, 0 ) += ( this->GetPoint( i ).Y() - DeltaPosition(i,1)  ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 1, 1 ) += ( this->GetPoint( i ).Y() - DeltaPosition(i,1)  ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 2, 0 ) += ( this->GetPoint( i ).Z() - DeltaPosition(i,2)  ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 2, 1 ) += ( this->GetPoint( i ).Z() - DeltaPosition(i,2)  ) * ( shape_functions_gradients[pnt]( i, 1 ) );
}

rResult[pnt] = jacobian;
}

return rResult;
}



Matrix& Jacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const override
{
if (rResult.size1() != 3 || rResult.size2() != 2)
rResult.resize( 3, 2 , false);
rResult.clear();
ShapeFunctionsGradientsType shape_functions_gradients =
CalculateShapeFunctionsIntegrationPointsLocalGradients( ThisMethod );
Matrix ShapeFunctionsGradientInIntegrationPoint =
shape_functions_gradients( IntegrationPointIndex );
DenseVector<double> ShapeFunctionsValuesInIntegrationPoint = ZeroVector( 8 );

ShapeFunctionsValuesInIntegrationPoint = row(
CalculateShapeFunctionsIntegrationPointsValues( ThisMethod ),
IntegrationPointIndex );


for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
rResult( 0, 0 ) +=
( this->GetPoint( i ).X() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 0 ) );
rResult( 0, 1 ) +=
( this->GetPoint( i ).X() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 1 ) );
rResult( 1, 0 ) +=
( this->GetPoint( i ).Y() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 0 ) );
rResult( 1, 1 ) +=
( this->GetPoint( i ).Y() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 1 ) );
rResult( 2, 0 ) +=
( this->GetPoint( i ).Z() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 0 ) );
rResult( 2, 1 ) +=
( this->GetPoint( i ).Z() ) * ( ShapeFunctionsGradientInIntegrationPoint( i, 1 ) );
}

return rResult;
}



Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
if (rResult.size1() != 3 || rResult.size2() != 2)
rResult.resize( 3, 2 , false);
rResult.clear();
Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(
shape_functions_gradients, rPoint );

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
rResult( 0, 0 ) += ( this->GetPoint( i ).X() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 0, 1 ) += ( this->GetPoint( i ).X() ) * ( shape_functions_gradients( i, 1 ) );
rResult( 1, 0 ) += ( this->GetPoint( i ).Y() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 1, 1 ) += ( this->GetPoint( i ).Y() ) * ( shape_functions_gradients( i, 1 ) );
rResult( 2, 0 ) += ( this->GetPoint( i ).Z() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 2, 1 ) += ( this->GetPoint( i ).Z() ) * ( shape_functions_gradients( i, 1 ) );
}

return rResult;
}


SizeType EdgesNumber() const override
{
return 4;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ), this->pGetPoint( 4 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 1 ), this->pGetPoint( 2 ), this->pGetPoint( 5 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 2 ), this->pGetPoint( 3 ), this->pGetPoint( 6 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 3 ), this->pGetPoint( 0 ), this->pGetPoint( 7 ) ) );
return edges;
}





double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return -(( 1.0 - rPoint[0] )*( 1.0 - rPoint[1] )
*( 1.0 + rPoint[0]
+ rPoint[1] ) ) / 4.0;
case 1:
return -(( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1] )*( 1.0
- rPoint[0] + rPoint[1] ) ) / 4.0;
case 2 :
return -(( 1.0 + rPoint[0] )
*( 1.0 + rPoint[1] )*( 1.0
- rPoint[0] - rPoint[1] ) ) / 4.0;
case 3 :
return -(( 1.0 - rPoint[0] )*( 1.0
+ rPoint[1] )*( 1.0 )*( 1.0
+ rPoint[0] - rPoint[1] ) ) / 4.0;
case 4 :
return (( 1.0 -rPoint[0]*rPoint[0] )
*( 1.0 - rPoint[1] ) ) / 2.0;
case 5 :
return (( 1.0 + rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] ) ) / 2.0 ;
case 6 :
return (( 1.0 -rPoint[0]
*rPoint[0] )*( 1.0 + rPoint[1] ) ) / 2.0 ;
case 7 :
return (( 1.0 -rPoint[0] )
*( 1.0 - rPoint[1]*rPoint[1] ) ) / 2.0 ;
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 8) rResult.resize(8,false);

rResult[0] =   -(( 1.0 - rCoordinates[0] )*( 1.0 - rCoordinates[1] )
*( 1.0 + rCoordinates[0]
+ rCoordinates[1] ) ) / 4.0;
rResult[1] =    -(( 1.0 + rCoordinates[0] )
*( 1.0 - rCoordinates[1] )*( 1.0
- rCoordinates[0] + rCoordinates[1] ) ) / 4.0;
rResult[2] =    -(( 1.0 + rCoordinates[0] )
*( 1.0 + rCoordinates[1] )*( 1.0
- rCoordinates[0] - rCoordinates[1] ) ) / 4.0;
rResult[3] =    -(( 1.0 - rCoordinates[0] )*( 1.0
+ rCoordinates[1] )*( 1.0 )*( 1.0
+ rCoordinates[0] - rCoordinates[1] ) ) / 4.0;
rResult[4] =    (( 1.0 -rCoordinates[0]*rCoordinates[0] )
*( 1.0 - rCoordinates[1] ) ) / 2.0;
rResult[5] =    (( 1.0 + rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] ) ) / 2.0 ;
rResult[6] =    (( 1.0 -rCoordinates[0]
*rCoordinates[0] )*( 1.0 + rCoordinates[1] ) ) / 2.0 ;
rResult[7] =    (( 1.0 -rCoordinates[0] )
*( 1.0 - rCoordinates[1]*rCoordinates[1] ) ) / 2.0 ;

return rResult;
}





std::string Info() const override
{
return "2 dimensional quadrilateral with eight nodes in 2D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "2 dimensional quadrilateral with eight nodes in 2D space";
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
rResult.resize( 8, 2, false );
noalias( rResult ) = ZeroMatrix( 8, 2 );

rResult( 0, 0 ) = (( -1.0 + rPoint[1] ) * ( -2.0 ) * ( 1.0 + 2.0
* rPoint[0] + rPoint[1] - 1.0 ) ) / 8.0;
rResult( 0, 1 ) = (( -1.0 + rPoint[0] ) * ( -2.0 ) * ( 1.0 + rPoint[0] + 2.0
* rPoint[1] - 1.0 ) ) / 8.0;

rResult( 1, 0 ) = -(( -1.0 + rPoint[1] ) * ( -2.0 ) * ( 1.0 - 2.0
* rPoint[0] + rPoint[1] - 1.0 ) ) / 8.0;
rResult( 1, 1 ) = (( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[0] - 2.0
* rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0;

rResult( 2, 0 ) = -(( 1.0 + rPoint[1] ) * ( -1.0 + 2.0
* rPoint[0] + rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0;
rResult( 2, 1 ) = -(( 1.0 + rPoint[0] ) * ( -1.0 + rPoint[0] + 2.0
* rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0;

rResult( 3, 0 ) = (( 1.0 + rPoint[1] ) * ( -1.0 - 2.0
* rPoint[0] + rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0;
rResult( 3, 1 ) = -(( -1.0 + rPoint[0] ) * ( -2.0 ) * ( 1.0 + rPoint[0] - 2.0
* rPoint[1] - 1.0 ) ) / 8.0;

rResult( 4, 0 ) = -( rPoint[0] * ( -1.0 + rPoint[1] ) * ( -2.0 ) ) / 2.0;
rResult( 4, 1 ) = -(( -1.0 + rPoint[0] * rPoint[0] ) * ( -2.0 ) ) / 4.0;

rResult( 5, 0 ) = (( -1.0 + rPoint[1] * rPoint[1] ) * ( -2.0 ) ) / 4.0;
rResult( 5, 1 ) = (( 1.0 + rPoint[0] ) * rPoint[1] * ( -2.0 ) ) / 2.0;

rResult( 6, 0 ) = ( rPoint[0] * ( 1.0 + rPoint[1] ) * ( -2.0 ) ) / 2.0;
rResult( 6, 1 ) = (( -1.0 + rPoint[0] * rPoint[0] ) * ( -2.0 ) ) / 4.0;

rResult( 7, 0 ) = -(( -1.0 + rPoint[1] * rPoint[1] ) * ( -2.0 ) ) / 4.0;
rResult( 7, 1 ) = -(( -1.0 + rPoint[0] ) * rPoint[1] * ( -2.0 ) ) / 2.0;

return( rResult );
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
rResult.resize( 8, 2, false );
noalias( rResult ) = ZeroMatrix( 8, 2 );
rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) = -1.0;
rResult( 2, 0 ) =  1.0;
rResult( 2, 1 ) =  1.0;
rResult( 3, 0 ) = -1.0;
rResult( 3, 1 ) =  1.0;
rResult( 4, 0 ) =  0.0;
rResult( 4, 1 ) = -1.0;
rResult( 5, 0 ) =  1.0;
rResult( 5, 1 ) =  0.0;
rResult( 6, 0 ) =  0.0;
rResult( 6, 1 ) =  1.0;
rResult( 7, 0 ) = -1.0;
rResult( 7, 1 ) =  0.0;
return rResult;
}


virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, PointType& rPoint )
{
rResult.resize( 8, 2, false );
noalias( rResult ) = ZeroMatrix( 8, 2 );

rResult( 0, 0 ) = (( -1.0 + rPoint.Y() ) * ( -2.0 ) * ( 1.0 + 2.0
* rPoint.X() + rPoint.Y() - 1.0 ) ) / 8.0;
rResult( 0, 1 ) = (( -1.0 + rPoint.X() ) * ( -2.0 ) * ( 1.0 + rPoint.X() + 2.0
* rPoint.Y() - 1.0 ) ) / 8.0;

rResult( 1, 0 ) = -(( -1.0 + rPoint.Y() ) * ( -2.0 ) * ( 1.0 - 2.0
* rPoint.X() + rPoint.Y() - 1.0 ) ) / 8.0;
rResult( 1, 1 ) = (( 1.0 + rPoint.X() ) * ( -1.0 + rPoint.X() - 2.0
* rPoint.Y() + 1.0 ) * ( -2.0 ) ) / 8.0;

rResult( 2, 0 ) = -(( 1.0 + rPoint.Y() ) * ( -1.0 + 2.0
* rPoint.X() + rPoint.Y() + 1.0 ) * ( -2.0 ) ) / 8.0;
rResult( 2, 1 ) = -(( 1.0 + rPoint.X() ) * ( -1.0 + rPoint.X() + 2.0
* rPoint.Y() + 1.0 ) * ( -2.0 ) ) / 8.0;

rResult( 3, 0 ) = (( 1.0 + rPoint.Y() ) * ( -1.0 - 2.0
* rPoint.X() + rPoint.Y() + 1.0 ) * ( -2.0 ) ) / 8.0;
rResult( 3, 1 ) = -(( -1.0 + rPoint.X() ) * ( -2.0 ) * ( 1.0 + rPoint.X() - 2.0
* rPoint.Y() - 1.0 ) ) / 8.0;

rResult( 4, 0 ) = -( rPoint.X() * ( -1.0 + rPoint.Y() ) * ( -2.0 ) ) / 2.0;
rResult( 4, 1 ) = -(( -1.0 + rPoint.X() * rPoint.X() ) * ( -2.0 ) ) / 4.0;

rResult( 5, 0 ) = (( -1.0 + rPoint.Y() * rPoint.Y() ) * ( -2.0 ) ) / 4.0;
rResult( 5, 1 ) = (( 1.0 + rPoint.X() ) * rPoint.Y() * ( -2.0 ) ) / 2.0;

rResult( 6, 0 ) = ( rPoint.X() * ( 1.0 + rPoint.Y() ) * ( -2.0 ) ) / 2.0;
rResult( 6, 1 ) = (( -1.0 + rPoint.X() * rPoint.X() ) * ( -2.0 ) ) / 4.0;

rResult( 7, 0 ) = -(( -1.0 + rPoint.Y() * rPoint.Y() ) * ( -2.0 ) ) / 4.0;
rResult( 7, 1 ) = -(( -1.0 + rPoint.X() ) * rPoint.Y() * ( -2.0 ) ) / 2.0;

return rResult;
}


ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives( ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsGradientsType temp( this->PointsNumber() );
rResult.swap( temp );
}

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
rResult[i].resize( 2, 2, false );
noalias( rResult[i] ) = ZeroMatrix( 2, 2 );
}

rResult[0]( 0, 0 ) = ( 4.0 - 4 * rPoint[1] ) / 8.0;

rResult[0]( 0, 1 ) = ( -2.0 ) * ( 1.0 + 2.0 * rPoint[0] + rPoint[1] - 1.0 ) / 8.0
+ (( -1.0 + rPoint[1] ) * ( -2.0 ) ) / 8.0;
rResult[0]( 1, 0 ) = ( -2.0 ) * ( 1.0 + rPoint[0] + 2.0 * rPoint[1] - 1.0 ) / 8.0
+ (( -1.0 + rPoint[0] ) * ( -2.0 ) ) / 8.0;
rResult[0]( 1, 1 ) = (( -1.0 + rPoint[0] ) * ( -2.0 ) * ( 2.0 ) ) / 8.0;

rResult[1]( 0, 0 ) = -(( -1.0 + rPoint[1] ) * ( -2.0 ) * ( -2.0 ) ) / 8.0;
rResult[1]( 0, 1 ) = ( 2.0 ) * ( 1.0 - 2.0 * rPoint[0] + rPoint[1] - 1.0 ) / 8.0
- (( -1.0 + rPoint[1] ) * ( -2.0 ) ) / 8.0;
rResult[1]( 1, 0 ) = ( -1.0 + rPoint[0] - 2.0 * rPoint[1] + 1.0 ) * ( -2.0 ) / 8.0
+ (( 1.0 + rPoint[0] ) * ( -2.0 ) ) / 8.0;
rResult[1]( 1, 1 ) = (( 1.0 + rPoint[0] ) * ( -2.0 ) * ( -2.0 ) ) / 8.0;

rResult[2]( 0, 0 ) = -(( 1.0 + rPoint[1] ) * ( 2.0 ) * ( -2.0 ) ) / 8.0;
rResult[2]( 0, 1 ) = -(( 1.0 ) * ( -1.0 + 2.0 * rPoint[0] + rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0
- (( 1.0 + rPoint[1] ) * ( -2.0 ) ) / 8.0;
rResult[2]( 1, 0 ) = -(( 1.0 ) * ( -1.0 + rPoint[0] + 2.0 * rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0
- (( 1.0 + rPoint[0] ) * ( -2.0 ) ) / 8.0;
rResult[2]( 1, 1 ) = -(( 1.0 + rPoint[0] ) * ( 2.0 ) * ( -2.0 ) ) / 8.0;

rResult[3]( 0, 0 ) = (( 1.0 + rPoint[1] ) * ( -2.0 ) * ( -2.0 ) ) / 8.0;
rResult[3]( 0, 1 ) = (( 1.0 ) * ( -1.0 - 2.0 * rPoint[0] + rPoint[1] + 1.0 ) * ( -2.0 ) ) / 8.0
+ (( 1.0 + rPoint[1] ) * ( -2.0 ) ) / 8.0;
rResult[3]( 1, 0 ) = -(( -2.0 ) * ( 1.0 + rPoint[0] - 2.0 * rPoint[1] - 1.0 ) ) / 8.0
- (( -1.0 + rPoint[0] ) * ( -2.0 ) ) / 8.0;
rResult[3]( 1, 1 ) = -(( -1.0 + rPoint[0] ) * ( -2.0 ) * ( -2.0 ) ) / 8.0;

rResult[4]( 0, 0 ) = -(( -1.0 + rPoint[1] ) * ( -2.0 ) ) / 2.0;
rResult[4]( 0, 1 ) = -( rPoint[0] * ( -2.0 ) ) / 2.0;
rResult[4]( 1, 0 ) = -(( 2.0 * rPoint[0] ) * ( -2.0 ) ) / 4.0;
rResult[4]( 1, 1 ) = 0.0;

rResult[5]( 0, 0 ) = 0.0;
rResult[5]( 0, 1 ) = (( 2.0 * rPoint[1] ) * ( -2.0 ) ) / 4.0;
rResult[5]( 1, 0 ) = ( rPoint[1] * ( -2.0 ) ) / 2.0;
rResult[5]( 1, 1 ) = (( 1.0 + rPoint[0] ) * ( -2.0 ) ) / 2.0;

rResult[6]( 0, 0 ) = (( 1.0 + rPoint[1] ) * ( -2.0 ) ) / 2.0;
rResult[6]( 0, 1 ) = ( rPoint[0] * ( -2.0 ) ) / 2.0;
rResult[6]( 1, 0 ) = (( 2.0 * rPoint[0] ) * ( -2.0 ) ) / 4.0;
rResult[6]( 1, 1 ) = 0.0;

rResult[7]( 0, 0 ) = 0.0;
rResult[7]( 0, 1 ) = -(( 2.0 * rPoint[1] ) * ( -2.0 ) ) / 4.0;
rResult[7]( 1, 0 ) = -( rPoint[1] * ( -2.0 ) ) / 2.0;
rResult[7]( 1, 1 ) = -(( -1.0 + rPoint[0] ) * ( -2.0 ) ) / 2.0;

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

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
for ( int j = 0; j < 2; j++ )
{
rResult[i][j].resize( 2, 2, false );
noalias( rResult[i][j] ) = ZeroMatrix( 2, 2 );
}
}

rResult[0][0]( 0, 0 ) = 0.0;
rResult[0][0]( 0, 1 ) = -0.5;
rResult[0][0]( 1, 0 ) = -0.5;
rResult[0][0]( 1, 1 ) = -0.5;
rResult[0][1]( 0, 0 ) = -0.5;
rResult[0][1]( 0, 1 ) = -0.5;
rResult[0][1]( 1, 0 ) = -0.5;
rResult[0][1]( 1, 1 ) = 0.0;
rResult[1][0]( 0, 0 ) = 0.0;
rResult[1][0]( 0, 1 ) = -0.5;
rResult[1][0]( 1, 0 ) = -0.5;
rResult[1][0]( 1, 1 ) = 0.5;
rResult[1][1]( 0, 0 ) = -0.5;
rResult[1][1]( 0, 1 ) = 0.5;
rResult[1][1]( 1, 0 ) = 0.5;
rResult[1][1]( 1, 1 ) = 0.0;
rResult[2][0]( 0, 0 ) = 0.0;
rResult[2][0]( 0, 1 ) = 0.5;
rResult[2][0]( 1, 0 ) = 0.5;
rResult[2][0]( 1, 1 ) = 0.5;
rResult[2][1]( 0, 0 ) = 0.5;
rResult[2][1]( 0, 1 ) = 0.5;
rResult[2][1]( 1, 0 ) = 0.5;
rResult[2][1]( 1, 1 ) = 0.0;
rResult[3][0]( 0, 0 ) = 0.0;
rResult[3][0]( 0, 1 ) = 0.5;
rResult[3][0]( 1, 0 ) = 0.5;
rResult[3][0]( 1, 1 ) = -0.5;
rResult[3][1]( 0, 0 ) = 0.5;
rResult[3][1]( 0, 1 ) = -0.5;
rResult[3][1]( 1, 0 ) = -0.5;
rResult[3][1]( 1, 1 ) = 0.0;
rResult[4][0]( 0, 0 ) = 0.0;
rResult[4][0]( 0, 1 ) = 1.0;
rResult[4][0]( 1, 0 ) = 1.0;
rResult[4][0]( 1, 1 ) = 0.0;
rResult[4][1]( 0, 0 ) = 1.0;
rResult[4][1]( 0, 1 ) = 0.0;
rResult[4][1]( 1, 0 ) = 0.0;
rResult[4][1]( 1, 1 ) = 0.0;
rResult[5][0]( 0, 0 ) = 0.0;
rResult[5][0]( 0, 1 ) = 0.0;
rResult[5][0]( 1, 0 ) = 0.0;
rResult[5][0]( 1, 1 ) = -1.0;
rResult[5][1]( 0, 0 ) = 0.0;
rResult[5][1]( 0, 1 ) = -1.0;
rResult[5][1]( 1, 0 ) = 1.0;
rResult[5][1]( 1, 1 ) = 0.0;
rResult[6][0]( 0, 0 ) = 0.0;
rResult[6][0]( 0, 1 ) = -1.0;
rResult[6][0]( 1, 0 ) = -1.0;
rResult[6][0]( 1, 1 ) = 0.0;
rResult[6][1]( 0, 0 ) = -1.0;
rResult[6][1]( 0, 1 ) = 0.0;
rResult[6][1]( 1, 0 ) = 0.0;
rResult[6][1]( 1, 1 ) = 0.0;
rResult[7][0]( 0, 0 ) = 0.0;
rResult[7][0]( 0, 1 ) = 0.0;
rResult[7][0]( 1, 0 ) = 0.0;
rResult[7][0]( 1, 1 ) = 1.0;
rResult[7][1]( 0, 0 ) = 0.0;
rResult[7][1]( 0, 1 ) = 1.0;
rResult[7][1]( 1, 0 ) = -1.0;
rResult[7][1]( 1, 1 ) = 0.0;

return rResult;
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

Quadrilateral3D8(): BaseType( PointsArrayType(), &msGeometryData ) {}







static Matrix CalculateShapeFunctionsIntegrationPointsValues( typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 8;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
row( shape_function_values, pnt )[0] =
-(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() )
* ( 1.0 + integration_points[pnt].X()
+ integration_points[pnt].Y() ) ) / 4.0;
row( shape_function_values, pnt )[1] =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) * ( 1.0
- integration_points[pnt].X()
+ integration_points[pnt].Y() ) ) / 4.0;
row( shape_function_values, pnt )[2] =
-(( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) * ( 1.0
- integration_points[pnt].X()
- integration_points[pnt].Y() ) ) / 4.0;
row( shape_function_values, pnt )[3] =
-(( 1.0 - integration_points[pnt].X() ) * ( 1.0
+ integration_points[pnt].Y() ) * ( 1.0 ) * ( 1.0
+ integration_points[pnt].X()
- integration_points[pnt].Y() ) ) / 4.0;
row( shape_function_values, pnt )[4] =
(( 1.0 - integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() ) ) / 2.0;
row( shape_function_values, pnt )[5] =
(( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 2.0 ;
row( shape_function_values, pnt )[6] =
(( 1.0 - integration_points[pnt].X()
* integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() ) ) / 2.0 ;
row( shape_function_values, pnt )[7] =
(( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y()
* integration_points[pnt].Y() ) ) / 2.0 ;
}

return shape_function_values;
}



static ShapeFunctionsGradientsType
CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Matrix result = ZeroMatrix( 8, 2 );

result( 0, 0 ) = (( -1.0 + integration_points[pnt].Y() )
* ( -2.0 ) * ( 1.0 + 2.0
* integration_points[pnt].X()
+ integration_points[pnt].Y() - 1.0 ) ) / 8.0;
result( 0, 1 ) = (( -1.0 + integration_points[pnt].X() ) * ( -2.0 ) * ( 1.0
+ integration_points[pnt].X() + 2.0
* integration_points[pnt].Y() - 1.0 ) ) / 8.0;

result( 1, 0 ) = -(( -1.0 + integration_points[pnt].Y() )
* ( -2.0 ) * ( 1.0 - 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y() - 1.0 ) ) / 8.0;
result( 1, 1 ) = (( 1.0 + integration_points[pnt].X() ) * ( -1.0
+ integration_points[pnt].X() - 2.0
* integration_points[pnt].Y() + 1.0 ) * ( -2.0 ) ) / 8.0;

result( 2, 0 ) = -(( 1.0 + integration_points[pnt].Y() )
* ( -1.0 + 2.0 * integration_points[pnt].X()
+ integration_points[pnt].Y() + 1.0 ) * ( -2.0 ) ) / 8.0;
result( 2, 1 ) = -(( 1.0 + integration_points[pnt].X() ) * ( -1.0
+ integration_points[pnt].X() + 2.0
* integration_points[pnt].Y() + 1.0 ) * ( -2.0 ) ) / 8.0;

result( 3, 0 ) = (( 1.0 + integration_points[pnt].Y() ) * ( -1.0 - 2.0
* integration_points[pnt].X()
+ integration_points[pnt].Y() + 1.0 ) * ( -2.0 ) ) / 8.0;
result( 3, 1 ) = -(( -1.0 + integration_points[pnt].X() ) * ( -2.0 ) * ( 1.0
+ integration_points[pnt].X() - 2.0
* integration_points[pnt].Y() - 1.0 ) ) / 8.0;

result( 4, 0 ) = -( integration_points[pnt].X() * ( -1.0
+ integration_points[pnt].Y() ) * ( -2.0 ) ) / 2.0;

result( 4, 1 ) = -(( -1.0
+ integration_points[pnt].X()
* integration_points[pnt].X() ) * ( -2.0 ) ) / 4.0;

result( 5, 0 ) = (( -1.0
+ integration_points[pnt].Y()
* integration_points[pnt].Y() ) * ( -2.0 ) ) / 4.0;

result( 5, 1 ) = (( 1.0
+ integration_points[pnt].X() )
* integration_points[pnt].Y() * ( -2.0 ) ) / 2.0;

result( 6, 0 ) = ( integration_points[pnt].X() * ( 1.0
+ integration_points[pnt].Y() ) * ( -2.0 ) ) / 2.0;
result( 6, 1 ) = (( -1.0
+ integration_points[pnt].X()
* integration_points[pnt].X() ) * ( -2.0 ) ) / 4.0;

result( 7, 0 ) = -(( -1.0
+ integration_points[pnt].Y()
* integration_points[pnt].Y() ) * ( -2.0 ) ) / 4.0;

result( 7, 1 ) = -(( -1.0
+ integration_points[pnt].X() )
* integration_points[pnt].Y() * ( -2.0 ) ) / 2.0;

d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}


static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < QuadrilateralGaussLegendreIntegrationPoints1,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralGaussLegendreIntegrationPoints2,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralGaussLegendreIntegrationPoints3,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralGaussLegendreIntegrationPoints4,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralGaussLegendreIntegrationPoints5,
2, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}


static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral3D8<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}



template<class TOtherPointType> friend class Quadrilateral3D8;



}; 



template< class TPointType > inline std::istream& operator >> (
std::istream& rIStream,
Quadrilateral3D8<TPointType>& rThis );


template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const Quadrilateral3D8<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}

template<class TPointType>
const GeometryData Quadrilateral3D8<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_3,
Quadrilateral3D8<TPointType>::AllIntegrationPoints(),
Quadrilateral3D8<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType>
const GeometryDimension Quadrilateral3D8<TPointType>::msGeometryDimension(3, 2);

}  
