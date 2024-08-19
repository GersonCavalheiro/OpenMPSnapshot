
#pragma once



#include "geometries/line_2d_2.h"
#include "geometries/triangle_2d_3.h"
#include "utilities/integration_utilities.h"
#include "integration/quadrilateral_gauss_legendre_integration_points.h"
#include "integration/quadrilateral_collocation_integration_points.h"

namespace Kratos
{






template<class TPointType> class Quadrilateral2D4
: public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;


typedef Line2D2<TPointType> EdgeType;


KRATOS_CLASS_POINTER_DEFINITION( Quadrilateral2D4 );


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



Quadrilateral2D4( typename PointType::Pointer pFirstPoint,
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

explicit Quadrilateral2D4( const PointsArrayType& rThisPoints )
: BaseType( rThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 4 )
KRATOS_ERROR << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral2D4(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 4 ) << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}

explicit Quadrilateral2D4(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 4) << "Invalid points number. Expected 4, given " << this->PointsNumber() << std::endl;
}


Quadrilateral2D4( Quadrilateral2D4 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> explicit Quadrilateral2D4( Quadrilateral2D4<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Quadrilateral2D4() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Quadrilateral;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Quadrilateral2D4;
}



Quadrilateral2D4& operator=( const Quadrilateral2D4& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Quadrilateral2D4& operator=( Quadrilateral2D4<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );
return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Quadrilateral2D4( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Quadrilateral2D4( NewGeometryId, rGeometry.Points() ) );
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
if (rResult.size1() != 4 || rResult.size2() != 2)
rResult.resize(4, 2, false);
rResult(0, 0) = -1.0;
rResult(0, 1) = -1.0;
rResult(1, 0) = 1.0;
rResult(1, 1) = -1.0;
rResult(2, 0) = 1.0;
rResult(2, 1) = 1.0;
rResult(3, 0) = -1.0;
rResult(3, 1) = 1.0;
return rResult;
}




double Length() const override
{
double length = 0.000;
length = sqrt( fabs( Area() ) );
return length;

}


double Area() const override
{
return IntegrationUtilities::ComputeArea2DGeometry(*this);
}


double Volume() const override
{
KRATOS_WARNING("Quadrilateral2D4") << "Method not well defined. Replace with DomainSize() instead. This method preserves current behaviour but will be changed in June 2023 (returning error instead)" << std::endl;
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
this->PointLocalCoordinates( rResult, rPoint );

if ( std::abs(rResult[0]) <= (1.0+Tolerance) )
{
if ( std::abs(rResult[1]) <= (1.0+Tolerance) )
{
return true;
}
}

return false;
}



SizeType EdgesNumber() const override
{
return 4;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 1 ), this->pGetPoint( 2 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 2 ), this->pGetPoint( 3 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 3 ), this->pGetPoint( 0 ) ) );
return edges;
}


bool HasIntersection( const Point& rLowPoint, const Point& rHighPoint ) const override
{
Triangle2D3<PointType> triangle_0 (this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 2 )
);
Triangle2D3<PointType> triangle_1 (this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 )
);

if      ( triangle_0.HasIntersection(rLowPoint, rHighPoint) ) return true;
else if ( triangle_1.HasIntersection(rLowPoint, rHighPoint) ) return true;
else return false;
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
return "2 dimensional quadrilateral with four nodes in 2D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "2 dimensional quadrilateral with four nodes in 2D space";
}



void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
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
rResult.resize( 4, 2 , false);
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




virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, PointType& rPoint )
{
rResult.resize( 4, 2 , false);
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


ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives( ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const override
{
if ( rResult.size() != this->PointsNumber() )
{
ShapeFunctionsGradientsType temp( this->PointsNumber() );
rResult.swap( temp );
}

rResult[0].resize( 2, 2 , false);
rResult[1].resize( 2, 2 , false);
rResult[2].resize( 2, 2 , false);
rResult[3].resize( 2, 2 , false);

rResult[0]( 0, 0 ) = 0.0;
rResult[0]( 0, 1 ) = 0.25;
rResult[0]( 1, 0 ) = 0.25;
rResult[0]( 1, 1 ) = 0.0;

rResult[1]( 0, 0 ) =  0.0;
rResult[1]( 0, 1 ) = -0.25;
rResult[1]( 1, 0 ) = -0.25;
rResult[1]( 1, 1 ) =  0.0;

rResult[2]( 0, 0 ) = 0.0;
rResult[2]( 0, 1 ) = 0.25;
rResult[2]( 1, 0 ) = 0.25;
rResult[2]( 1, 1 ) = 0.0;

rResult[3]( 0, 0 ) =  0.0;
rResult[3]( 0, 1 ) = -0.25;
rResult[3]( 1, 0 ) = -0.25;
rResult[3]( 1, 1 ) =  0.0;

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
for ( unsigned int j = 0; j < 2; j++ )
{
rResult[i][j].resize( 2, 2 , false);
noalias( rResult[i][j] ) = ZeroMatrix( 2, 2 );
}
}


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

Quadrilateral2D4(): BaseType( PointsArrayType(), &msGeometryData ) {}






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
Matrix result( 4, 2 );
result( 0, 0 ) = -0.25 * ( 1.0 - integration_points[pnt].Y() );
result( 0, 1 ) = -0.25 * ( 1.0 - integration_points[pnt].X() );
result( 1, 0 ) = 0.25 * ( 1.0 - integration_points[pnt].Y() );
result( 1, 1 ) = -0.25 * ( 1.0 + integration_points[pnt].X() );
result( 2, 0 ) = 0.25 * ( 1.0 + integration_points[pnt].Y() );
result( 2, 1 ) = 0.25 * ( 1.0 + integration_points[pnt].X() );
result( 3, 0 ) = -0.25 * ( 1.0 + integration_points[pnt].Y() );
result( 3, 1 ) = 0.25 * ( 1.0 - integration_points[pnt].X() );
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
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Quadrilateral2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 ),
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Quadrilateral2D4;



}; 




template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream,
Quadrilateral2D4<TPointType>& rThis );

template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const Quadrilateral2D4<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}


template<class TPointType> const
GeometryData Quadrilateral2D4<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Quadrilateral2D4<TPointType>::AllIntegrationPoints(),
Quadrilateral2D4<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Quadrilateral2D4<TPointType>::msGeometryDimension(2, 2);

}