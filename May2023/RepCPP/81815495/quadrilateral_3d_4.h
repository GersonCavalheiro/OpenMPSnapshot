
#pragma once



#include "geometries/line_3d_2.h"
#include "geometries/triangle_3d_3.h"
#include "utilities/integration_utilities.h"
#include "integration/quadrilateral_gauss_legendre_integration_points.h"
#include "integration/quadrilateral_collocation_integration_points.h"
#include "utilities/geometrical_projection_utilities.h"

namespace Kratos
{






template<class TPointType> class Quadrilateral3D4
: public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;

typedef Geometry<TPointType> GeometryType;


typedef Line3D2<TPointType> EdgeType;


typedef Quadrilateral3D4<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Quadrilateral3D4 );


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




Quadrilateral3D4( typename PointType::Pointer pFirstPoint,
typename PointType::Pointer pSecondPoint,
typename PointType::Pointer pThirdPoint,
typename PointType::Pointer pFourthPoint )
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( pFirstPoint );
this->Points().push_back( pSecondPoint );
this->Points().push_back( pThirdPoint );
this->Points().push_back( pFourthPoint );
}

explicit Quadrilateral3D4( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 4 )
KRATOS_ERROR << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral3D4(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 4 ) << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral3D4(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 4) << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}


Quadrilateral3D4( Quadrilateral3D4 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> explicit Quadrilateral3D4( Quadrilateral3D4<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Quadrilateral3D4() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Quadrilateral;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Quadrilateral3D4;
}



Quadrilateral3D4& operator=( const Quadrilateral3D4& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Quadrilateral3D4& operator=( Quadrilateral3D4<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );
return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Quadrilateral3D4( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Quadrilateral3D4( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}

SizeType PointsNumberInDirection(IndexType LocalDirectionIndex) const override
{
if ((LocalDirectionIndex == 0) || (LocalDirectionIndex == 1)) {
return 2;
}
KRATOS_ERROR << "Possible direction index reaches from 0-1. Given direction index: "
<< LocalDirectionIndex << std::endl;
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
rResult.resize( 4, 2, false );
noalias( rResult ) = ZeroMatrix( 4, 2 );
rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) = -1.0;
rResult( 2, 0 ) =  1.0;
rResult( 2, 1 ) =  1.0;
rResult( 3, 0 ) = -1.0;
rResult( 3, 1 ) =  1.0;
return rResult;
}




double Length() const override
{
return std::sqrt( Area() );
}


double Area() const override
{
const IntegrationMethod integration_method = msGeometryData.DefaultIntegrationMethod();
return IntegrationUtilities::ComputeDomainSize(*this, integration_method);
}


double Volume() const override
{
KRATOS_WARNING("Quadrilateral3D4") << "Method not well defined. Replace with DomainSize() instead. This method preserves current behaviour but will be changed in June 2023 (returning error instead)" << std::endl;
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
PointLocalCoordinatesImplementation( rResult, rPoint, true );

if ( std::abs(rResult[0]) <= (1.0+Tolerance) ) {
if ( std::abs(rResult[1]) <= (1.0+Tolerance) ) {
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
return PointLocalCoordinatesImplementation(rResult, rPoint);
}



JacobiansType& Jacobian(
JacobiansType& rResult,
IntegrationMethod ThisMethod
) const override
{
const ShapeFunctionsGradientsType& shape_functions_gradients =
msGeometryData.ShapeFunctionsLocalGradients( ThisMethod );
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
jacobian( 0, 0 ) +=
( this->GetPoint( i ).X() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 0, 1 ) +=
( this->GetPoint( i ).X() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 1, 0 ) +=
( this->GetPoint( i ).Y() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 1, 1 ) +=
( this->GetPoint( i ).Y() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 2, 0 ) +=
( this->GetPoint( i ).Z() ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 2, 1 ) +=
( this->GetPoint( i ).Z() ) * ( shape_functions_gradients[pnt]( i, 1 ) );
}

rResult[pnt] = jacobian;
} 

return rResult;
}


JacobiansType& Jacobian(
JacobiansType& rResult,
IntegrationMethod ThisMethod,
Matrix & DeltaPosition
) const override
{
const ShapeFunctionsGradientsType& shape_functions_gradients =
msGeometryData.ShapeFunctionsLocalGradients( ThisMethod );
Matrix shape_functions_values = CalculateShapeFunctionsIntegrationPointsValues( ThisMethod );

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
jacobian( 0, 0 ) +=
( this->GetPoint( i ).X() - DeltaPosition(i,0) ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 0, 1 ) +=
( this->GetPoint( i ).X() - DeltaPosition(i,0) ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 1, 0 ) +=
( this->GetPoint( i ).Y() - DeltaPosition(i,1) ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 1, 1 ) +=
( this->GetPoint( i ).Y() - DeltaPosition(i,1) ) * ( shape_functions_gradients[pnt]( i, 1 ) );
jacobian( 2, 0 ) +=
( this->GetPoint( i ).Z() - DeltaPosition(i,2) ) * ( shape_functions_gradients[pnt]( i, 0 ) );
jacobian( 2, 1 ) +=
( this->GetPoint( i ).Z() - DeltaPosition(i,2) ) * ( shape_functions_gradients[pnt]( i, 1 ) );
}

rResult[pnt] = jacobian;
}

return rResult;
}


Matrix& Jacobian(
Matrix& rResult,
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod
) const override
{
if (rResult.size1() != 3 || rResult.size2() != 2 )
rResult.resize( 3, 2, false );
noalias(rResult) = ZeroMatrix(3, 2);
Matrix shape_functions_gradients = msGeometryData.ShapeFunctionLocalGradient(IntegrationPointIndex, ThisMethod );

for ( unsigned int i = 0; i < this->PointsNumber(); i++ )
{
rResult( 0, 0 ) +=
( this->GetPoint( i ).X() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 0, 1 ) +=
( this->GetPoint( i ).X() ) * ( shape_functions_gradients( i, 1 ) );
rResult( 1, 0 ) +=
( this->GetPoint( i ).Y() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 1, 1 ) +=
( this->GetPoint( i ).Y() ) * ( shape_functions_gradients( i, 1 ) );
rResult( 2, 0 ) +=
( this->GetPoint( i ).Z() ) * ( shape_functions_gradients( i, 0 ) );
rResult( 2, 1 ) +=
( this->GetPoint( i ).Z() ) * ( shape_functions_gradients( i, 1 ) );
}

return rResult;
}



Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
if (rResult.size1() != 3 || rResult.size2() != 2 )
rResult.resize( 3, 2, false );
noalias(rResult) = ZeroMatrix(3, 2);

Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(shape_functions_gradients, rPoint );

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



Vector& DeterminantOfJacobian(
Vector& rResult,
IntegrationMethod ThisMethod
) const override
{
const unsigned int integration_points_number = msGeometryData.IntegrationPointsNumber( ThisMethod );
if(rResult.size() != integration_points_number)
{
rResult.resize(integration_points_number,false);
}

JacobiansType jacobian;
this->Jacobian( jacobian, ThisMethod);

for ( unsigned int pnt = 0; pnt < integration_points_number; pnt++ )
{
const double det_j = std::pow(jacobian[pnt](0,1),2)*(std::pow(jacobian[pnt](1,0),2) + std::pow(jacobian[pnt](2,0),2)) + std::pow(jacobian[pnt](1,1)*jacobian[pnt](2,0) - jacobian[pnt](1,0)*jacobian[pnt](2,1),2) - 2.0*jacobian[pnt](0,0)*jacobian[pnt](0,1)*(jacobian[pnt](1,0)*jacobian[pnt](1,1) + jacobian[pnt](2,0)*jacobian[pnt](2,1)) + std::pow(jacobian[pnt](0,0),2)*(std::pow(jacobian[pnt](1,1),2) + std::pow(jacobian[pnt](2,1),2));

if (det_j < 0.0) KRATOS_ERROR << "WARNING::NEGATIVE VALUE: NOT POSSIBLE TO EVALUATE THE JACOBIAN DETERMINANT" << std::endl;

rResult[pnt] = std::sqrt(det_j);
}

return rResult;
}



double DeterminantOfJacobian(
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod
) const override
{
Matrix jacobian( 3, 2 );

this->Jacobian( jacobian, IntegrationPointIndex, ThisMethod);

const double det_j = std::pow(jacobian(0,1),2)*(std::pow(jacobian(1,0),2) + std::pow(jacobian(2,0),2)) + std::pow(jacobian(1,1)*jacobian(2,0) - jacobian(1,0)*jacobian(2,1),2) - 2.0*jacobian(0,0)*jacobian(0,1)*(jacobian(1,0)*jacobian(1,1) + jacobian(2,0)*jacobian(2,1)) + std::pow(jacobian(0,0),2)*(std::pow(jacobian(1,1),2) + std::pow(jacobian(2,1),2));

if (det_j < 0.0) KRATOS_ERROR << "WARNING::NEGATIVE VALUE: NOT POSSIBLE TO EVALUATE THE JACOBIAN DETERMINANT" << std::endl;

return std::sqrt(det_j);
}




double DeterminantOfJacobian( const CoordinatesArrayType& rPoint ) const override
{
Matrix jacobian( 3, 2 );

this->Jacobian( jacobian, rPoint);

const double det_j = std::pow(jacobian(0,1),2)*(std::pow(jacobian(1,0),2) + std::pow(jacobian(2,0),2)) + std::pow(jacobian(1,1)*jacobian(2,0) - jacobian(1,0)*jacobian(2,1),2) - 2.0*jacobian(0,0)*jacobian(0,1)*(jacobian(1,0)*jacobian(1,1) + jacobian(2,0)*jacobian(2,1)) + std::pow(jacobian(0,0),2)*(std::pow(jacobian(1,1),2) + std::pow(jacobian(2,1),2));

if (det_j < 0.0) KRATOS_ERROR << "WARNING::NEGATIVE VALUE: NOT POSSIBLE TO EVALUATE THE JACOBIAN DETERMINANT" << std::endl;

return std::sqrt(det_j);
}



SizeType EdgesNumber() const override
{
return 4;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer EdgePointerType;
edges.push_back( EdgePointerType( new EdgeType( this->pGetPoint( 0 ), this->pGetPoint( 1 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType( this->pGetPoint( 1 ), this->pGetPoint( 2 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType( this->pGetPoint( 2 ), this->pGetPoint( 3 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType( this->pGetPoint( 3 ), this->pGetPoint( 0 ) ) ) );
return edges;
}



SizeType FacesNumber() const override
{
return 1;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();

faces.push_back( Kratos::make_shared<FaceType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ), this->pGetPoint( 2 ), this->pGetPoint( 3 )) );
return faces;
}

void NumberNodesInFaces (DenseVector<unsigned int>& NumberNodesInFaces) const override
{
if(NumberNodesInFaces.size() != 4 )
NumberNodesInFaces.resize(4,false);

NumberNodesInFaces[0]=2;
NumberNodesInFaces[1]=2;
NumberNodesInFaces[2]=2;
NumberNodesInFaces[3]=2;

}

void NodesInFaces (DenseMatrix<unsigned int>& NodesInFaces) const override
{
if(NodesInFaces.size1() != 3 || NodesInFaces.size2() != 4)
NodesInFaces.resize(3,4,false);
NodesInFaces(0,0)=0;
NodesInFaces(1,0)=2;
NodesInFaces(2,0)=3;
NodesInFaces(0,1)=1;
NodesInFaces(1,1)=3;
NodesInFaces(2,1)=0;
NodesInFaces(0,2)=2;
NodesInFaces(1,2)=0;
NodesInFaces(2,2)=1;
NodesInFaces(0,3)=3;
NodesInFaces(1,3)=1;
NodesInFaces(2,3)=2;
}


bool HasIntersection(const GeometryType& ThisGeometry) const override
{
Triangle3D3<PointType> triangle_0 (this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 2 )
);
Triangle3D3<PointType> triangle_1 (this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 )
);
Triangle3D3<PointType> triangle_2 (ThisGeometry.pGetPoint( 0 ),
ThisGeometry.pGetPoint( 1 ),
ThisGeometry.pGetPoint( 2 )
);
Triangle3D3<PointType> triangle_3 (ThisGeometry.pGetPoint( 2 ),
ThisGeometry.pGetPoint( 3 ),
ThisGeometry.pGetPoint( 0 )
);

if      ( triangle_0.HasIntersection(triangle_2) ) return true;
else if ( triangle_1.HasIntersection(triangle_2) ) return true;
else if ( triangle_0.HasIntersection(triangle_3) ) return true;
else if ( triangle_1.HasIntersection(triangle_3) ) return true;
else return false;
}


bool HasIntersection( const Point& rLowPoint, const Point& rHighPoint ) const override
{
Triangle3D3<PointType> triangle_0 (this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 2 )
);
Triangle3D3<PointType> triangle_1 (this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 )
);

if      ( triangle_0.HasIntersection(rLowPoint, rHighPoint) ) return true;
else if ( triangle_1.HasIntersection(rLowPoint, rHighPoint) ) return true;
else return false;
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
return( 0.25*( 1.0 - rPoint[0] )*( 1.0 - rPoint[1] ) );
case 1:
return( 0.25*( 1.0 + rPoint[0] )*( 1.0 - rPoint[1] ) );
case 2:
return( 0.25*( 1.0 + rPoint[0] )*( 1.0 + rPoint[1] ) );
case 3:
return( 0.25*( 1.0 - rPoint[0] )*( 1.0 + rPoint[1] ) );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 4) rResult.resize(4,false);
rResult[0] =  0.25*( 1.0 - rCoordinates[0] )*( 1.0 - rCoordinates[1] );
rResult[1] =  0.25*( 1.0 + rCoordinates[0] )*( 1.0 - rCoordinates[1] );
rResult[2] =  0.25*( 1.0 + rCoordinates[0] )*( 1.0 + rCoordinates[1] );
rResult[3] =  0.25*( 1.0 - rCoordinates[0] )*( 1.0 + rCoordinates[1] );

return rResult;
}



std::string Info() const override
{
return "2 dimensional quadrilateral with four nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << Info();
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
const ShapeFunctionsGradientsType& shape_function_local_gradient
= msGeometryData.ShapeFunctionsLocalGradients( ThisMethod );
const int& integration_points_number
= msGeometryData.IntegrationPointsNumber( ThisMethod );
ShapeFunctionsGradientsType Result( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Result[pnt] = shape_function_local_gradient[pnt];
}

return Result;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients()
{
IntegrationMethod ThisMethod = msGeometryData.DefaultIntegrationMethod();
const ShapeFunctionsGradientsType& shape_function_local_gradient
= msGeometryData.ShapeFunctionsLocalGradients( ThisMethod );
const int integration_points_number
= msGeometryData.IntegrationPointsNumber( ThisMethod );
ShapeFunctionsGradientsType Result( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Result[pnt] = shape_function_local_gradient[pnt];
}

return Result;
}


Matrix& ShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rPoint
) const override
{
rResult.resize( 4, 2, false );
noalias( rResult ) = ZeroMatrix( 4, 2 );
rResult( 0, 0 ) = -0.25 * ( 1.0 - rPoint[1] );
rResult( 0, 1 ) = -0.25 * ( 1.0 - rPoint[0] );
rResult( 1, 0 ) =  0.25 * ( 1.0 - rPoint[1] );
rResult( 1, 1 ) = -0.25 * ( 1.0 + rPoint[0] );
rResult( 2, 0 ) =  0.25 * ( 1.0 + rPoint[1] );
rResult( 2, 1 ) =  0.25 * ( 1.0 + rPoint[0] );
rResult( 3, 0 ) = -0.25 * ( 1.0 + rPoint[1] );
rResult( 3, 1 ) =  0.25 * ( 1.0 - rPoint[0] );
return rResult;
}


virtual Matrix& ShapeFunctionsGradients(
Matrix& rResult,
PointType& rPoint
)
{
rResult.resize( 4, 2, false );
rResult( 0, 0 ) = -0.25 * ( 1.0 - rPoint.Y() );
rResult( 0, 1 ) = -0.25 * ( 1.0 - rPoint.X() );
rResult( 1, 0 ) =  0.25 * ( 1.0 - rPoint.Y() );
rResult( 1, 1 ) = -0.25 * ( 1.0 + rPoint.X() );
rResult( 2, 0 ) =  0.25 * ( 1.0 + rPoint.Y() );
rResult( 2, 1 ) =  0.25 * ( 1.0 + rPoint.X() );
rResult( 3, 0 ) = -0.25 * ( 1.0 + rPoint.Y() );
rResult( 3, 1 ) =  0.25 * ( 1.0 - rPoint.X() );
return rResult;
}


ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives(
ShapeFunctionsSecondDerivativesType& rResult,
const CoordinatesArrayType& rPoint
) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsGradientsType temp( this->PointsNumber() );
rResult.swap( temp );
}

rResult[0].resize( 2, 2, false );
rResult[1].resize( 2, 2, false );
rResult[2].resize( 2, 2, false );
rResult[3].resize( 2, 2, false );

rResult[0]( 0, 0 ) = 0.0;
rResult[0]( 0, 1 ) = 0.25;
rResult[0]( 1, 0 ) = 0.25;
rResult[0]( 1, 1 ) = 0.0;
rResult[1]( 0, 0 ) = 0.0;
rResult[1]( 0, 1 ) = -0.25;
rResult[1]( 1, 0 ) = -0.25;
rResult[1]( 1, 1 ) = 0.0;
rResult[2]( 0, 0 ) = 0.0;
rResult[2]( 0, 1 ) = 0.25;
rResult[2]( 1, 0 ) = 0.25;
rResult[2]( 1, 1 ) = 0.0;
rResult[3]( 0, 0 ) = 0.0;
rResult[3]( 0, 1 ) = -0.25;
rResult[3]( 1, 0 ) = -0.25;
rResult[3]( 1, 1 ) = 0.0;
return rResult;
}


ShapeFunctionsThirdDerivativesType& ShapeFunctionsThirdDerivatives(
ShapeFunctionsThirdDerivativesType& rResult,
const CoordinatesArrayType& rPoint
) const override
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

rResult[0][0].resize( 2, 2, false );
rResult[0][1].resize( 2, 2, false );
rResult[1][0].resize( 2, 2, false );
rResult[1][1].resize( 2, 2, false );
rResult[2][0].resize( 2, 2, false );
rResult[2][1].resize( 2, 2, false );
rResult[3][0].resize( 2, 2, false );
rResult[3][1].resize( 2, 2, false );

for ( int i = 0; i < 4; i++ )
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

const int result = ProjectionPointGlobalToLocalSpace(rPointGlobalCoordinates, rProjectedPointLocalCoordinates, Tolerance);

this->GlobalCoordinates(rProjectedPointGlobalCoordinates, rProjectedPointLocalCoordinates);

return result;
}

int ProjectionPointLocalToLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
CoordinatesArrayType pt_gl_coords;
this->GlobalCoordinates(pt_gl_coords, rPointLocalCoordinates);

return this->ProjectionPointGlobalToLocalSpace(pt_gl_coords, rProjectionPointLocalCoordinates, Tolerance);
}

int ProjectionPointGlobalToLocalSpace(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
const std::size_t max_number_of_iterations = 10;

CoordinatesArrayType proj_pt_gl_coords = this->Center();
array_1d<double, 3> normal = this->UnitNormal(proj_pt_gl_coords);

double distance;
std::size_t iter;

for (iter = 0; iter < max_number_of_iterations; ++iter) {
proj_pt_gl_coords = GeometricalProjectionUtilities::FastProject<CoordinatesArrayType>(
proj_pt_gl_coords,
rPointGlobalCoordinates,
normal,
distance);

if (norm_2(this->UnitNormal(proj_pt_gl_coords) - normal) < Tolerance) {
break;
}

noalias(normal) = this->UnitNormal(proj_pt_gl_coords);
}

PointLocalCoordinates(rProjectionPointLocalCoordinates, proj_pt_gl_coords);

if (iter >= max_number_of_iterations - 1) {
return 0;
} else {
return 1;
}
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

Quadrilateral3D4(): BaseType( PointsArrayType(), &msGeometryData ) {}







CoordinatesArrayType& PointLocalCoordinatesImplementation(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint,
const bool IsInside = false
) const
{
BoundedMatrix<double,3,4> X;
BoundedMatrix<double,3,2> DN;
for(IndexType i=0; i<this->size();i++) {
X(0, i) = this->GetPoint( i ).X();
X(1, i) = this->GetPoint( i ).Y();
X(2, i) = this->GetPoint( i ).Z();
}

static constexpr double MaxNormPointLocalCoordinates = 300.0;
static constexpr std::size_t MaxIteratioNumberPointLocalCoordinates = 500;
static constexpr double MaxTolerancePointLocalCoordinates = 1.0e-8;

Matrix J = ZeroMatrix( 2, 2 );
Matrix invJ = ZeroMatrix( 2, 2 );

rResult = ZeroVector( 3 );
array_1d<double, 2> DeltaXi = ZeroVector( 2 );
const array_1d<double, 3> zero_array = ZeroVector(3);
array_1d<double, 3> CurrentGlobalCoords;

for ( IndexType k = 0; k < MaxIteratioNumberPointLocalCoordinates; k++ ) {
noalias(CurrentGlobalCoords) = zero_array;
this->GlobalCoordinates( CurrentGlobalCoords, rResult );

noalias( CurrentGlobalCoords ) = rPoint - CurrentGlobalCoords;

Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(shape_functions_gradients, rResult );
noalias(DN) = prod(X,shape_functions_gradients);

noalias(J) = prod(trans(DN),DN);
const array_1d<double, 2> res = prod(trans(DN), CurrentGlobalCoords);

const double det_j = J( 0, 0 ) * J( 1, 1 ) - J( 0, 1 ) * J( 1, 0 );

invJ( 0, 0 ) = ( J( 1, 1 ) ) / ( det_j );
invJ( 1, 0 ) = -( J( 1, 0 ) ) / ( det_j );
invJ( 0, 1 ) = -( J( 0, 1 ) ) / ( det_j );
invJ( 1, 1 ) = ( J( 0, 0 ) ) / ( det_j );

DeltaXi( 0 ) = invJ( 0, 0 ) * res[0] + invJ( 0, 1 ) * res[1];
DeltaXi( 1 ) = invJ( 1, 0 ) * res[0] + invJ( 1, 1 ) * res[1];

rResult[0] += DeltaXi[0];
rResult[1] += DeltaXi[1];

if ( norm_2( DeltaXi ) > MaxNormPointLocalCoordinates ) {
KRATOS_WARNING_IF("Quadrilateral3D4", IsInside == false && k > 0) << "detJ =\t" << det_j << " DeltaX =\t" << DeltaXi << " stopping calculation. Iteration:\t" << k << std::endl;
break;
}

if ( norm_2( DeltaXi ) < MaxTolerancePointLocalCoordinates )
break;
}

return rResult;
}


static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 4;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
shape_function_values( pnt, 0 ) =
0.25 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
shape_function_values( pnt, 1 ) =
0.25 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 - integration_points[pnt].Y() );
shape_function_values( pnt, 2 ) =
0.25 * ( 1.0 + integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
shape_function_values( pnt, 3 ) =
0.25 * ( 1.0 - integration_points[pnt].X() )
* ( 1.0 + integration_points[pnt].Y() );
}

return shape_function_values;
}


static ShapeFunctionsGradientsType CalculateShapeFunctionsIntegrationPointsLocalGradients( typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
Matrix result( 4, 2 );
result( 0, 0 ) = -0.25 * ( 1.0 - integration_points[pnt].Y() );
result( 0, 1 ) = -0.25 * ( 1.0 - integration_points[pnt].X() );
result( 1, 0 ) =  0.25 * ( 1.0 - integration_points[pnt].Y() );
result( 1, 1 ) = -0.25 * ( 1.0 + integration_points[pnt].X() );
result( 2, 0 ) =  0.25 * ( 1.0 + integration_points[pnt].Y() );
result( 2, 1 ) =  0.25 * ( 1.0 + integration_points[pnt].X() );
result( 3, 0 ) = -0.25 * ( 1.0 + integration_points[pnt].Y() );
result( 3, 1 ) =  0.25 * ( 1.0 - integration_points[pnt].X() );
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
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralCollocationIntegrationPoints1,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralCollocationIntegrationPoints2,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralCollocationIntegrationPoints3,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralCollocationIntegrationPoints4,
2, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < QuadrilateralCollocationIntegrationPoints5,
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
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Quadrilateral3D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Quadrilateral3D4;



}; 




template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream,
Quadrilateral3D4<TPointType>& rThis );

template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const Quadrilateral3D4<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}


template<class TPointType>
const GeometryData Quadrilateral3D4<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Quadrilateral3D4<TPointType>::AllIntegrationPoints(),
Quadrilateral3D4<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType>
const GeometryDimension Quadrilateral3D4<TPointType>::msGeometryDimension(3, 2);

}
