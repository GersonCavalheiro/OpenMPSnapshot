
#pragma once



#include "geometries/line_2d_3.h"
#include "integration/triangle_gauss_legendre_integration_points.h"

namespace Kratos
{






template<class TPointType>
class Triangle2D6
: public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;


typedef Line2D3<TPointType> EdgeType;


KRATOS_CLASS_POINTER_DEFINITION( Triangle2D6 );


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



Triangle2D6( typename PointType::Pointer pFirstPoint,
typename PointType::Pointer pSecondPoint,
typename PointType::Pointer pThirdPoint,
typename PointType::Pointer pFourthPoint,
typename PointType::Pointer pFifthPoint,
typename PointType::Pointer pSixthPoint
)
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( pFirstPoint );
this->Points().push_back( pSecondPoint );
this->Points().push_back( pThirdPoint );
this->Points().push_back( pFourthPoint );
this->Points().push_back( pFifthPoint );
this->Points().push_back( pSixthPoint );
}

Triangle2D6( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 6 )
{
KRATOS_ERROR << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}
}

explicit Triangle2D6(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 6 ) << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}

explicit Triangle2D6(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType( rGeometryName, rThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF(this->PointsNumber() != 6) << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}


Triangle2D6( Triangle2D6 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Triangle2D6( Triangle2D6<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}


~Triangle2D6() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Triangle;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Triangle2D6;
}



Triangle2D6& operator=( const Triangle2D6& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Triangle2D6& operator=( Triangle2D6<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );
return *this;
}



typename BaseType::Pointer Create(
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Triangle2D6( rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Triangle2D6( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Triangle2D6( rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Triangle2D6( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
rResult.resize( 6, 2,false );
noalias( rResult ) = ZeroMatrix( 6, 2 );
rResult( 0, 0 ) =  0.0;
rResult( 0, 1 ) =  0.0;
rResult( 1, 0 ) =  1.0;
rResult( 1, 1 ) =  0.0;
rResult( 2, 0 ) =  0.0;
rResult( 2, 1 ) =  1.0;
rResult( 3, 0 ) =  0.5;
rResult( 3, 1 ) =  0.0;
rResult( 4, 0 ) =  0.5;
rResult( 4, 1 ) =  0.5;
rResult( 5, 0 ) =  0.0;
rResult( 5, 1 ) =  0.5;
return rResult;
}




double Length() const override
{
return std::sqrt( std::abs( Area() ) );
}


double Area() const override
{
return IntegrationUtilities::ComputeArea2DGeometry(*this);
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

if ( (rResult[0] >= (0.0-Tolerance)) && (rResult[0] <= (1.0+Tolerance)) )
{
if ( (rResult[1] >= (0.0-Tolerance)) && (rResult[1] <= (1.0+Tolerance)) )
{
if ( (rResult[0] + rResult[1]) <= (1.0+Tolerance) )
{
return true;
}
}
}

return false;
}



Vector& ShapeFunctionsValues(Vector& rResult, const CoordinatesArrayType& rCoordinates) const override
{
if (rResult.size() != 6)
rResult.resize(6, false);
const double thirdCoord = 1.0 - rCoordinates[0] - rCoordinates[1];
rResult[0] = thirdCoord * (2.0 * thirdCoord - 1.0);
rResult[1] = rCoordinates[0] * (2.0 * rCoordinates[0] - 1.0);
rResult[2] = rCoordinates[1] * (2.0 * rCoordinates[1] - 1.0);
rResult[3] = 4.0 * thirdCoord * rCoordinates[0];
rResult[4] = 4.0 * rCoordinates[0] * rCoordinates[1];
rResult[5] = 4.0 * rCoordinates[1] * thirdCoord;
return rResult;
}


double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{

double thirdCoord = 1 - rPoint[0] - rPoint[1];

switch ( ShapeFunctionIndex )
{
case 0:
return( thirdCoord*( 2*thirdCoord - 1 ) );
case 1:
return( rPoint[0]*( 2*rPoint[0] - 1 ) );
case 2:
return( rPoint[1]*( 2*rPoint[1] - 1 ) );
case 3:
return( 4*thirdCoord*rPoint[0] );
case 4:
return( 4*rPoint[0]*rPoint[1] );
case 5:
return( 4*rPoint[1]*thirdCoord );

default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}



std::string Info() const override
{
return "2 dimensional triangle with six nodes in 2D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "2 dimensional triangle with six nodes in 2D space";
}



void PrintData( std::ostream& rOStream ) const override
{
PrintInfo( rOStream );
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
rOStream << "    Jacobian in the origin\t : " << jacobian;
}



SizeType EdgesNumber() const override
{
return 3;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();

edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ), this->pGetPoint( 3 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 1 ), this->pGetPoint( 2 ), this->pGetPoint( 4 ) ) );
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 2 ), this->pGetPoint( 0 ), this->pGetPoint( 5 ) ) );
return edges;
}

SizeType FacesNumber() const override
{
return 3;
}

void NumberNodesInFaces (DenseVector<unsigned int>& NumberNodesInFaces) const override
{
if(NumberNodesInFaces.size() != 3 )
NumberNodesInFaces.resize(3,false);

NumberNodesInFaces[0]=3;
NumberNodesInFaces[1]=3;
NumberNodesInFaces[2]=3;

}


void NodesInFaces (DenseMatrix<unsigned int>& NodesInFaces) const override
{
if(NodesInFaces.size1() != 4 || NodesInFaces.size2() != 3)
NodesInFaces.resize(4,3,false);

NodesInFaces(0,0)=0;
NodesInFaces(1,0)=1;
NodesInFaces(2,0)=4;
NodesInFaces(3,0)=2;

NodesInFaces(0,1)=1;
NodesInFaces(1,1)=2;
NodesInFaces(2,1)=5;
NodesInFaces(3,1)=0;

NodesInFaces(0,2)=2;
NodesInFaces(1,2)=0;
NodesInFaces(2,2)=3;
NodesInFaces(3,2)=1;

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
rResult.resize( 6, 2 ,false);
double thirdCoord = 1 - rPoint[0] - rPoint[1];
double thirdCoord_DX = -1;
double thirdCoord_DY = -1;

noalias( rResult ) = ZeroMatrix( 6, 2 );
rResult( 0, 0 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DX;
rResult( 0, 1 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DY;
rResult( 1, 0 ) =  4 * rPoint[0] - 1;
rResult( 1, 1 ) =  0;
rResult( 2, 0 ) =  0;
rResult( 2, 1 ) =  4 * rPoint[1] - 1;
rResult( 3, 0 ) =  4 * thirdCoord_DX * rPoint[0] + 4 * thirdCoord;
rResult( 3, 1 ) =  4 * thirdCoord_DY * rPoint[0];
rResult( 4, 0 ) =  4 * rPoint[1];
rResult( 4, 1 ) =  4 * rPoint[0];
rResult( 5, 0 ) =  4 * rPoint[1] * thirdCoord_DX;
rResult( 5, 1 ) =  4 * rPoint[1] * thirdCoord_DY + 4 * thirdCoord;
return rResult;
}




virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, CoordinatesArrayType& rPoint )
{
rResult.resize( 6, 2 ,false);
double thirdCoord = 1 - rPoint[0] - rPoint[1];
double thirdCoord_DX = -1;
double thirdCoord_DY = -1;

noalias( rResult ) = ZeroMatrix( 6, 2 );
rResult( 0, 0 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DX;
rResult( 0, 1 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DY;
rResult( 1, 0 ) =  4 * rPoint[0] - 1;
rResult( 1, 1 ) =  0;
rResult( 2, 0 ) =  0;
rResult( 2, 1 ) =  4 * rPoint[1] - 1;
rResult( 3, 0 ) =  4 * thirdCoord_DX * rPoint[0] + 4 * thirdCoord;
rResult( 3, 1 ) =  4 * thirdCoord_DY * rPoint[0];
rResult( 4, 0 ) =  4 * rPoint[1];
rResult( 4, 1 ) =  4 * rPoint[0];
rResult( 5, 0 ) =  4 * rPoint[1] * thirdCoord_DX;
rResult( 5, 1 ) =  4 * rPoint[1] * thirdCoord_DY + 4 * thirdCoord;
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
rResult[3].resize( 2, 2 ,false);
rResult[4].resize( 2, 2 ,false);
rResult[5].resize( 2, 2 ,false);

rResult[0]( 0, 0 ) = 4.0;
rResult[0]( 0, 1 ) = 4.0;
rResult[0]( 1, 0 ) = 4.0;
rResult[0]( 1, 1 ) = 4.0;
rResult[1]( 0, 0 ) = 4.0;
rResult[1]( 0, 1 ) = 0.0;
rResult[1]( 1, 0 ) = 0.0;
rResult[1]( 1, 1 ) = 0.0;
rResult[2]( 0, 0 ) = 0.0;
rResult[2]( 0, 1 ) = 0.0;
rResult[2]( 1, 0 ) = 0.0;
rResult[2]( 1, 1 ) = 4.0;
rResult[3]( 0, 0 ) = -8.0;
rResult[3]( 0, 1 ) = -4.0;
rResult[3]( 1, 0 ) = -4.0;
rResult[3]( 1, 1 ) = 0.0;
rResult[4]( 0, 0 ) = 0.0;
rResult[4]( 0, 1 ) = 4.0;
rResult[4]( 1, 0 ) = 4.0;
rResult[4]( 1, 1 ) = 0.0;
rResult[5]( 0, 0 ) = 0.0;
rResult[5]( 0, 1 ) = -4.0;
rResult[5]( 1, 0 ) = -4.0;
rResult[5]( 1, 1 ) = -8.0;

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
rResult[3][0].resize( 2, 2 ,false);
rResult[3][1].resize( 2, 2 ,false);
rResult[4][0].resize( 2, 2 ,false);
rResult[4][1].resize( 2, 2 ,false);
rResult[5][0].resize( 2, 2 ,false);
rResult[5][1].resize( 2, 2 ,false);


for ( int i = 0; i < 6; i++ )
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

Triangle2D6(): BaseType( PointsArrayType(), &msGeometryData ) {}







static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 6;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
double thirdCoord = 1 - integration_points[pnt].X() - integration_points[pnt].Y();

shape_function_values( pnt, 0 ) = thirdCoord * ( 2 * thirdCoord - 1 ) ;
shape_function_values( pnt, 1 ) = integration_points[pnt].X() * ( 2 * integration_points[pnt].X() - 1 ) ;
shape_function_values( pnt, 2 ) =  integration_points[pnt].Y() * ( 2 * integration_points[pnt].Y() - 1 ) ;
shape_function_values( pnt, 3 ) =  4 * thirdCoord * integration_points[pnt].X();
shape_function_values( pnt, 4 ) =  4 * integration_points[pnt].X() * integration_points[pnt].Y();
shape_function_values( pnt, 5 ) =  4 * integration_points[pnt].Y() * thirdCoord;

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
Matrix result( 6, 2 );
double thirdCoord = 1 - integration_points[pnt].X() - integration_points[pnt].Y();
double thirdCoord_DX = -1;
double thirdCoord_DY = -1;

noalias( result ) = ZeroMatrix( 6, 2 );
result( 0, 0 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DX;
result( 0, 1 ) = ( 4 * thirdCoord - 1 ) * thirdCoord_DY;
result( 1, 0 ) =  4 * integration_points[pnt].X() - 1;
result( 1, 1 ) =  0;
result( 2, 0 ) =  0;
result( 2, 1 ) =  4 * integration_points[pnt].Y() - 1;
result( 3, 0 ) =  4 * thirdCoord_DX * integration_points[pnt].X() + 4 * thirdCoord;
result( 3, 1 ) =  4 * thirdCoord_DY * integration_points[pnt].X();
result( 4, 0 ) =  4 * integration_points[pnt].Y();
result( 4, 1 ) =  4 * integration_points[pnt].X();
result( 5, 0 ) =  4 * integration_points[pnt].Y() * thirdCoord_DX;
result( 5, 1 ) =  4 * integration_points[pnt].Y() * thirdCoord_DY + 4 * thirdCoord;

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
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 )
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
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Triangle2D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 )
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Triangle2D6;



}; 




template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream,
Triangle2D6<TPointType>& rThis );

template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const Triangle2D6<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}


template<class TPointType> const
GeometryData Triangle2D6<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Triangle2D6<TPointType>::AllIntegrationPoints(),
Triangle2D6<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType>
const GeometryDimension Triangle2D6<TPointType>::msGeometryDimension(2, 2);

}
