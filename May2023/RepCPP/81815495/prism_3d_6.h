
#pragma once



#include "geometries/triangle_3d_3.h"
#include "geometries/quadrilateral_3d_4.h"
#include "utilities/integration_utilities.h"
#include "integration/prism_gauss_legendre_integration_points.h"

namespace Kratos
{

template<class TPointType> class Prism3D6 : public Geometry<TPointType>
{
public:



typedef Geometry<TPointType> BaseType;


typedef Line3D2<TPointType> EdgeType;
typedef Triangle3D3<TPointType> FaceType1;
typedef Quadrilateral3D4<TPointType> FaceType2;


KRATOS_CLASS_POINTER_DEFINITION( Prism3D6 );


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





Prism3D6( typename PointType::Pointer pPoint1,
typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3,
typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5,
typename PointType::Pointer pPoint6 )
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().reserve( 6 );
this->Points().push_back( pPoint1 );
this->Points().push_back( pPoint2 );
this->Points().push_back( pPoint3 );
this->Points().push_back( pPoint4 );
this->Points().push_back( pPoint5 );
this->Points().push_back( pPoint6 );
}

explicit Prism3D6( const PointsArrayType& rThisPoints )
: BaseType( rThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 6 ) << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}

explicit Prism3D6(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 6 ) << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}

explicit Prism3D6(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 6) << "Invalid points number. Expected 6, given " << this->PointsNumber() << std::endl;
}


Prism3D6( Prism3D6 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> explicit Prism3D6( Prism3D6<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Prism3D6() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Prism;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Prism3D6;
}




Prism3D6& operator=( const Prism3D6& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Prism3D6& operator=( Prism3D6<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}





typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Prism3D6( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Prism3D6( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}




double Length() const override
{
const double volume = Volume();

return std::pow(volume, 1.0/3.0)/3.0;
}


double Area() const override
{
return std::abs( this->DeterminantOfJacobian( PointType() ) ) * 0.5;
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
if ( rResult.size1() != 6 || rResult.size2() != 3 )
rResult.resize( 6, 3 ,false);

rResult( 0, 0 ) = 0.0;
rResult( 0, 1 ) = 0.0;
rResult( 0, 2 ) = 0.0;

rResult( 1, 0 ) = 1.0;
rResult( 1, 1 ) = 0.0;
rResult( 1, 2 ) = 0.0;

rResult( 2, 0 ) = 0.0;
rResult( 2, 1 ) = 1.0;
rResult( 2, 2 ) = 0.0;

rResult( 3, 0 ) = 0.0;
rResult( 3, 1 ) = 0.0;
rResult( 3, 2 ) = 1.0;

rResult( 4, 0 ) = 1.0;
rResult( 4, 1 ) = 0.0;
rResult( 4, 2 ) = 1.0;

rResult( 5, 0 ) = 0.0;
rResult( 5, 1 ) = 1.0;
rResult( 5, 2 ) = 1.0;

return rResult;
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( (rResult[0] >= (0.0 - Tolerance)) && (rResult[0] <= (1.0 + Tolerance)) )
if ( (rResult[1] >= (0.0 - Tolerance)) && (rResult[1] <= (1.0 + Tolerance)) )
if ( (rResult[2] >= (0.0 - Tolerance)) && (rResult[2] <= (1.0 + Tolerance)) )
if ((( 1.0 - ( rResult[0] + rResult[1] ) ) >= (0.0 - Tolerance) ) && (( 1.0 - ( rResult[0] + rResult[1] ) ) <= (1.0 + Tolerance) ) )
return true;

return false;
}



SizeType EdgesNumber() const override
{
return 9;
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
this->pGetPoint( 0 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 5 ),
this->pGetPoint( 3 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 3 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 5 ) ) ) );
return edges;
}



SizeType FacesNumber() const override
{
return 5;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer FacePointerType;
faces.push_back( FacePointerType( new FaceType1(
this->pGetPoint( 0 ),
this->pGetPoint( 2 ),
this->pGetPoint( 1 ) ) ) );
faces.push_back( FacePointerType( new FaceType1(
this->pGetPoint( 3 ),
this->pGetPoint( 4 ),
this->pGetPoint( 5 ) ) ) );
faces.push_back( FacePointerType( new FaceType2(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 5 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType2(
this->pGetPoint( 0 ),
this->pGetPoint( 3 ),
this->pGetPoint( 5 ),
this->pGetPoint( 2 ) ) ) );
faces.push_back( FacePointerType( new FaceType2(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 4 ),
this->pGetPoint( 3 ) ) ) );
return faces;
}

bool HasIntersection( const Point& rLowPoint, const Point& rHighPoint ) const override
{
if(FaceType1(this->pGetPoint(0),this->pGetPoint(2), this->pGetPoint(1)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(FaceType1(this->pGetPoint(3),this->pGetPoint(4), this->pGetPoint(5)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(FaceType2(this->pGetPoint(1),this->pGetPoint(2), this->pGetPoint(5), this->pGetPoint(4)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(FaceType2(this->pGetPoint(0),this->pGetPoint(3), this->pGetPoint(5), this->pGetPoint(2)).HasIntersection(rLowPoint, rHighPoint))
return true;
if(FaceType2(this->pGetPoint(0),this->pGetPoint(1), this->pGetPoint(4), this->pGetPoint(3)).HasIntersection(rLowPoint, rHighPoint))
return true;

CoordinatesArrayType local_coordinates;
if(IsInside(rLowPoint,local_coordinates))
return true;

return false;
}




double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return( 1.0 -( rPoint[0] + rPoint[1] + rPoint[2] - ( rPoint[0]*rPoint[2] ) - ( rPoint[1]*rPoint[2] ) ) );
case 1:
return( rPoint[0] - ( rPoint[0]*rPoint[2] ) );
case 2:
return( rPoint[1] - ( rPoint[1]*rPoint[2] ) );
case 3:
return( rPoint[2] - ( rPoint[0]*rPoint[2] ) - ( rPoint[1]*rPoint[2] ) );
case 4:
return( rPoint[0]*rPoint[2] );
case 5:
return( rPoint[1]*rPoint[2] );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this  << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (
Vector &rResult,
const CoordinatesArrayType& rCoordinates
) const override
{
if(rResult.size() != 6)
rResult.resize(6,false);

rResult[0] =  1.0 -( rCoordinates[0] + rCoordinates[1] + rCoordinates[2] - ( rCoordinates[0] *  rCoordinates[2] ) - ( rCoordinates[1] * rCoordinates[2] ) );
rResult[1] =  rCoordinates[0] - ( rCoordinates[0] * rCoordinates[2] );
rResult[2] =  rCoordinates[1] - ( rCoordinates[1] * rCoordinates[2] );
rResult[3] =  rCoordinates[2] - ( rCoordinates[0] * rCoordinates[2] ) - ( rCoordinates[1] * rCoordinates[2] );
rResult[4] =  rCoordinates[0] * rCoordinates[2];
rResult[5] =  rCoordinates[1] * rCoordinates[2];

return rResult;
}


Matrix& ShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rPoint
) const override
{
if(rResult.size1() != this->PointsNumber() || rResult.size2() != this->LocalSpaceDimension())
rResult.resize(this->PointsNumber(),this->LocalSpaceDimension(),false);

CalculateShapeFunctionsLocalGradients(rResult, rPoint);

return rResult;
}





std::string Info() const override
{
return "3 dimensional prism with six nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "3 dimensional prism with six nodes in 3D space";
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

Prism3D6(): BaseType( PointsArrayType(), &msGeometryData ) {}






static Matrix& CalculateShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rPoint
)
{
rResult( 0, 0 ) = -1.0 + rPoint[2];
rResult( 0, 1 ) = -1.0 + rPoint[2];
rResult( 0, 2 ) = -1.0 + rPoint[0] + rPoint[1];
rResult( 1, 0 ) =  1.0 - rPoint[2];
rResult( 1, 1 ) =  0.0;
rResult( 1, 2 ) = -rPoint[0];
rResult( 2, 0 ) =  0.0;
rResult( 2, 1 ) =  1.0 - rPoint[2];
rResult( 2, 2 ) = -rPoint[1];
rResult( 3, 0 ) = -rPoint[2];
rResult( 3, 1 ) = -rPoint[2];
rResult( 3, 2 ) =  1.0 - rPoint[0] - rPoint[1];
rResult( 4, 0 ) =  rPoint[2];
rResult( 4, 1 ) =  0.0;
rResult( 4, 2 ) =  rPoint[0];
rResult( 5, 0 ) =  0.0;
rResult( 5, 1 ) =  rPoint[2];
rResult( 5, 2 ) =  rPoint[1];
return rResult;
}



static Matrix CalculateShapeFunctionsIntegrationPointsValues(typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 6;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ ) {
shape_function_values( pnt, 0 ) = ( 1.0
- integration_points[pnt].X()
- integration_points[pnt].Y()
- integration_points[pnt].Z()
+ ( integration_points[pnt].X() * integration_points[pnt].Z() )
+ ( integration_points[pnt].Y() * integration_points[pnt].Z() ) );
shape_function_values( pnt, 1 ) = integration_points[pnt].X()
- ( integration_points[pnt].X() * integration_points[pnt].Z() );
shape_function_values( pnt, 2 ) = integration_points[pnt].Y()
- ( integration_points[pnt].Y() * integration_points[pnt].Z() );
shape_function_values( pnt, 3 ) = integration_points[pnt].Z()
- ( integration_points[pnt].X() * integration_points[pnt].Z() )
- ( integration_points[pnt].Y() * integration_points[pnt].Z() );
shape_function_values( pnt, 4 ) = ( integration_points[pnt].X() * integration_points[pnt].Z() );
shape_function_values( pnt, 5 ) = ( integration_points[pnt].Y() * integration_points[pnt].Z() );
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
Matrix result = ZeroMatrix( 6, 3 );
result( 0, 0 ) = -1.0 + integration_points[pnt].Z();
result( 0, 1 ) = -1.0 + integration_points[pnt].Z();
result( 0, 2 ) = -1.0 + integration_points[pnt].X() + integration_points[pnt].Y();
result( 1, 0 ) =  1.0 - integration_points[pnt].Z();
result( 1, 1 ) =  0.0;
result( 1, 2 ) =  -integration_points[pnt].X();
result( 2, 0 ) =  0.0;
result( 2, 1 ) =  1.0 - integration_points[pnt].Z();
result( 2, 2 ) =  -integration_points[pnt].Y();
result( 3, 0 ) =  -integration_points[pnt].Z();
result( 3, 1 ) =  -integration_points[pnt].Z();
result( 3, 2 ) =  1.0 - integration_points[pnt].X() - integration_points[pnt].Y();
result( 4, 0 ) =  integration_points[pnt].Z();
result( 4, 1 ) =  0.0;
result( 4, 2 ) =  integration_points[pnt].X();
result( 5, 0 ) =  0.0;
result( 5, 1 ) =  integration_points[pnt].Z();
result( 5, 2 ) =  integration_points[pnt].Y();
d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < PrismGaussLegendreIntegrationPoints1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPoints2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPoints3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPoints4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPoints5,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPointsExt1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPointsExt2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPointsExt3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPointsExt4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PrismGaussLegendreIntegrationPointsExt5,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Prism3D6<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Prism3D6;




};




template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream, Prism3D6<TPointType>& rThis );

template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream, const Prism3D6<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}

template<class TPointType> const
GeometryData Prism3D6<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Prism3D6<TPointType>::AllIntegrationPoints(),
Prism3D6<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Prism3D6<TPointType>::msGeometryDimension(3, 3);

}
