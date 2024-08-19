
#pragma once

#include <cmath> 


#include "includes/define.h"
#include "geometries/geometry.h"
#include "geometries/triangle_3d_3.h"
#include "geometries/quadrilateral_3d_4.h"
#include "integration/pyramid_gauss_legendre_integration_points.h"

namespace Kratos {



template<class TPointType>
class Pyramid3D5 : public Geometry<TPointType>
{
public:

typedef Geometry<TPointType> BaseType;


typedef Line3D2<TPointType> EdgeType;
typedef Triangle3D3<TPointType> FaceType1;
typedef Quadrilateral3D4<TPointType> FaceType2;

KRATOS_CLASS_POINTER_DEFINITION(Pyramid3D5);


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef TPointType PointType;


typedef typename BaseType::IndexType IndexType;


typedef typename BaseType::SizeType SizeType;


typedef typename BaseType::PointsArrayType PointsArrayType;


typedef typename BaseType::IntegrationPointType IntegrationPointType;


typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


typedef typename BaseType::IntegrationPointsContainerType IntegrationPointsContainerType;


typedef typename BaseType::ShapeFunctionsValuesContainerType ShapeFunctionsValuesContainerType;


typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef typename BaseType::ShapeFunctionsLocalGradientsContainerType
ShapeFunctionsLocalGradientsContainerType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;


typedef typename BaseType::GeometriesArrayType GeometriesArrayType;



explicit Pyramid3D5(
typename PointType::Pointer pPoint1,
typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3,
typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5)
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().reserve(5);
this->Points().push_back(pPoint1);
this->Points().push_back(pPoint2);
this->Points().push_back(pPoint3);
this->Points().push_back(pPoint4);
this->Points().push_back(pPoint5);
}

explicit Pyramid3D5( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 5 ) << "Invalid points number. Expected 5, given " << this->PointsNumber() << std::endl;
}

explicit Pyramid3D5(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 5 ) << "Invalid points number. Expected 5, given " << this->PointsNumber() << std::endl;
}

explicit Pyramid3D5(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 5) << "Invalid points number. Expected 5, given " << this->PointsNumber() << std::endl;
}


Pyramid3D5(Pyramid3D5 const& rOther)
: BaseType(rOther)
{
}


template<class TOtherPointType> Pyramid3D5(Pyramid3D5<TOtherPointType> const& rOther)
: BaseType(rOther)
{
}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Pyramid;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Pyramid3D5;
}



Pyramid3D5& operator=(const Pyramid3D5& rOther)
{
BaseType::operator=(rOther);

return *this;
}


template<class TOtherPointType>
Pyramid3D5& operator=(Pyramid3D5<TOtherPointType> const & rOther)
{
BaseType::operator=(rOther);

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Pyramid3D5( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Pyramid3D5( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}



SizeType EdgesNumber() const override
{
return 8;
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
this->pGetPoint( 0 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 4 ) ) ) );
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
this->pGetPoint( 1 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType1(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType2(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 3 ) ) ) );
faces.push_back( FacePointerType( new FaceType1(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType1(
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 4 ) ) ) );
return faces;
}


double Volume() const override
{
return IntegrationUtilities::ComputeVolume3DGeometry(*this);
}


double DomainSize() const override
{
return Volume();
}


int IsInsideLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
if ( std::abs( rPointLocalCoordinates[0] ) <= (1.0 + Tolerance) ) {
if ( std::abs( rPointLocalCoordinates[1] ) <= (1.0 + Tolerance) ) {
if ( std::abs( rPointLocalCoordinates[2] ) <= (1.0 + Tolerance) ) {
if ( (std::abs(rPointLocalCoordinates[0]) +
std::abs(rPointLocalCoordinates[1]) +
rPointLocalCoordinates[2]) <= (1.0 + Tolerance) ) {
return 1;
}
}
}
}

return 0;
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
if ( rResult.size1() != 5 || rResult.size2() != 3 )
rResult.resize( 5, 3, false );

rResult( 0, 0 ) = -1.0;
rResult( 0, 1 ) = -1.0;
rResult( 0, 2 ) = -1.0;

rResult( 1, 0 ) = +1.0;
rResult( 1, 1 ) = -1.0;
rResult( 1, 2 ) = -1.0;

rResult( 2, 0 ) = +1.0;
rResult( 2, 1 ) = +1.0;
rResult( 2, 2 ) = -1.0;

rResult( 3, 0 ) = -1.0;
rResult( 3, 1 ) = +1.0;
rResult( 3, 2 ) = -1.0;

rResult( 4, 0 ) =  0.0;
rResult( 4, 1 ) =  0.0;
rResult( 4, 2 ) = +1.0;

return rResult;
}




Vector& ShapeFunctionsValues(Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 5) rResult.resize(5,false);

for (std::size_t i=0; i<5; ++i) {
rResult[i] = ShapeFunctionValue(i, rCoordinates);
}

return rResult;
}


double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
return ShapeFunctionValueImpl(ShapeFunctionIndex, rPoint);
}



static Matrix CalculateShapeFunctionsIntegrationPointsValues(typename BaseType::IntegrationMethod ThisMethod)
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const std::size_t integration_points_number = integration_points.size();
const std::size_t points_number = 5;
Matrix shape_function_values( integration_points_number, points_number );

for (std::size_t pnt = 0; pnt<integration_points_number; ++pnt) {
for (std::size_t i=0; i<points_number; ++i) {
shape_function_values( pnt, i ) = ShapeFunctionValueImpl(i, integration_points[pnt]);
}
}

return shape_function_values;
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


static Matrix& CalculateShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rPoint
)
{
rResult.resize( 5, 3, false );
noalias( rResult ) = ZeroMatrix( 5, 3 );

rResult( 0, 0 ) =  (-0.125) * ( 1 - rPoint[1] ) * ( 1 - rPoint[2] ) ;
rResult( 0, 1 ) =  (-0.125) * ( 1 - rPoint[0] ) * ( 1 - rPoint[2] ) ;
rResult( 0, 2 ) =  (-0.125) * ( 1 - rPoint[0] ) * ( 1 - rPoint[1] ) ;

rResult( 1, 0 ) =  (+0.125) * ( 1 - rPoint[1] ) * ( 1 - rPoint[2] ) ;
rResult( 1, 1 ) =  (-0.125) * ( 1 + rPoint[0] ) * ( 1 - rPoint[2] ) ;
rResult( 1, 2 ) =  (-0.125) * ( 1 + rPoint[0] ) * ( 1 - rPoint[1] ) ;

rResult( 2, 0 ) =  (+0.125) * ( 1 + rPoint[1] ) * ( 1 - rPoint[2] ) ;
rResult( 2, 1 ) =  (+0.125) * ( 1 + rPoint[0] ) * ( 1 - rPoint[2] ) ;
rResult( 2, 2 ) =  (-0.125) * ( 1 + rPoint[0] ) * ( 1 + rPoint[1] ) ;

rResult( 3, 0 ) =  (-0.125) * ( 1 + rPoint[1] ) * ( 1 - rPoint[2] ) ;
rResult( 3, 1 ) =  (+0.125) * ( 1 - rPoint[0] ) * ( 1 - rPoint[2] ) ;
rResult( 3, 2 ) =  (-0.125) * ( 1 - rPoint[0] ) * ( 1 + rPoint[1] ) ;

rResult( 4, 0 ) =   0.00 ;
rResult( 4, 1 ) =   0.00 ;
rResult( 4, 2 ) =  +0.50 ;

return rResult;
}


static ShapeFunctionsGradientsType CalculateShapeFunctionsIntegrationPointsLocalGradients(typename BaseType::IntegrationMethod ThisMethod)
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)]; 

const std::size_t integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values(integration_points_number); 

Matrix result;

for (std::size_t pnt = 0; pnt<integration_points_number; ++pnt) {
d_shape_f_values[pnt] = CalculateShapeFunctionsLocalGradients(result, integration_points[pnt]);
}

return d_shape_f_values;
}



std::string Info() const override
{
return "3 dimensional pyramid with 5 nodes in 3D space";
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}


void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
}


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

Pyramid3D5() : BaseType( PointsArrayType(), &msGeometryData ) {}


static double ShapeFunctionValueImpl(
IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint)
{
switch ( ShapeFunctionIndex )
{
case 0:
return( (0.125) * (1 - rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[2]) );
case 1:
return( (0.125) * (1 + rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[2]) );
case 2:
return( (0.125) * (1 + rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[2]) );
case 3:
return( (0.125) * (1 - rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[2]) );
case 4:
return( (0.5) * (1 + rPoint[2]) );
default:
KRATOS_ERROR << "Wrong index of shape function:" << ShapeFunctionIndex  << std::endl;
}

return 0;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < PyramidGaussLegendreIntegrationPoints1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PyramidGaussLegendreIntegrationPoints2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PyramidGaussLegendreIntegrationPoints3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PyramidGaussLegendreIntegrationPoints4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < PyramidGaussLegendreIntegrationPoints5,
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
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_1),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_2),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_3),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_4),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients =
{
{
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_1),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_2),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_3),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_4),
Pyramid3D5<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Pyramid3D5;


}; 





template<class TPointType>
inline std::istream& operator >> (std::istream& rIStream,
Pyramid3D5<TPointType>& rThis);

template<class TPointType>
inline std::ostream& operator << (std::ostream& rOStream,
const Pyramid3D5<TPointType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

template<class TPointType> const
GeometryData Pyramid3D5<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Pyramid3D5<TPointType>::AllIntegrationPoints(),
Pyramid3D5<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Pyramid3D5<TPointType>::msGeometryDimension(3, 3);

}  
