
#pragma once

#include <cmath> 


#include "includes/define.h"
#include "geometries/geometry.h"
#include "utilities/integration_utilities.h"
#include "integration/pyramid_gauss_legendre_integration_points.h"

namespace Kratos {



template<class TPointType>
class Pyramid3D13 : public Geometry<TPointType>
{
public:

typedef Geometry<TPointType> BaseType;

KRATOS_CLASS_POINTER_DEFINITION(Pyramid3D13);


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



explicit Pyramid3D13(
typename PointType::Pointer pPoint1,
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
typename PointType::Pointer pPoint13)
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().reserve(13);
this->Points().push_back(pPoint1);
this->Points().push_back(pPoint2);
this->Points().push_back(pPoint3);
this->Points().push_back(pPoint4);
this->Points().push_back(pPoint5);
this->Points().push_back(pPoint6);
this->Points().push_back(pPoint7);
this->Points().push_back(pPoint8);
this->Points().push_back(pPoint9);
this->Points().push_back(pPoint10);
this->Points().push_back(pPoint11);
this->Points().push_back(pPoint12);
this->Points().push_back(pPoint13);
}

explicit Pyramid3D13( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
KRATOS_ERROR_IF( this->PointsNumber() != 13 ) << "Invalid points number. Expected 13, given " << this->PointsNumber() << std::endl;
}

explicit Pyramid3D13(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 13 ) << "Invalid points number. Expected 13, given " << this->PointsNumber() << std::endl;
}

explicit Pyramid3D13(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 13) << "Invalid points number. Expected 13, given " << this->PointsNumber() << std::endl;
}


Pyramid3D13(Pyramid3D13 const& rOther)
: BaseType(rOther)
{
}


template<class TOtherPointType> Pyramid3D13(Pyramid3D13<TOtherPointType> const& rOther)
: BaseType(rOther)
{
}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Pyramid;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Pyramid3D13;
}



Pyramid3D13& operator=(const Pyramid3D13& rOther)
{
BaseType::operator=(rOther);

return *this;
}


template<class TOtherPointType>
Pyramid3D13& operator=(Pyramid3D13<TOtherPointType> const & rOther)
{
BaseType::operator=(rOther);

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Pyramid3D13( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Pyramid3D13( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}



SizeType EdgesNumber() const override
{
return 8;
}


SizeType FacesNumber() const override
{
return 5;
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
if ( rResult.size1() != 13 || rResult.size2() != 3 )
rResult.resize( 13, 3, false );

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

rResult( 5, 0 ) =  0.0;
rResult( 5, 1 ) = -0.5;
rResult( 5, 2 ) = -1.0;

rResult( 6, 0 ) = +0.5;
rResult( 6, 1 ) =  0.0;
rResult( 6, 2 ) = -1.0;

rResult( 7, 0 ) =  0.0;
rResult( 7, 1 ) = +0.5;
rResult( 7, 2 ) = -1.0;

rResult( 8, 0 ) = +0.5;
rResult( 8, 1 ) =  0.0;
rResult( 8, 2 ) = -1.0;

rResult( 9, 0 ) = -0.5;
rResult( 9, 1 ) = -0.5;
rResult( 9, 2 ) =  0.0;

rResult( 10, 0 ) = +0.5;
rResult( 10, 1 ) = -0.5;
rResult( 10, 2 ) = 0.0;

rResult( 11, 0 ) = +0.5;
rResult( 11, 1 ) = +0.5;
rResult( 11, 2 ) = 0.0;

rResult( 12, 0 ) = -0.5;
rResult( 12, 1 ) = +0.5;
rResult( 12, 2 ) = 0.0;

return rResult;
}




Vector& ShapeFunctionsValues(Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 13) rResult.resize(13,false);

for (std::size_t i=0; i<13; ++i) {
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
const std::size_t points_number = 13;
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
rResult.resize( 13, 3, false );
noalias( rResult ) = ZeroMatrix( 13, 3 );

rResult( 0, 0 ) = (+0.0625) * (1 - rPoint[1]) * (1 - rPoint[2]) * (1 + 6*rPoint[0] + rPoint[1] + 4*rPoint[0]*rPoint[1] + rPoint[2] + 2*rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] + 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 0, 1 ) = (+0.0625) * (1 - rPoint[0]) * (1 - rPoint[2]) * (1 + rPoint[0] + 6*rPoint[1] + 4*rPoint[0]*rPoint[1] + rPoint[2] - rPoint[0]*rPoint[2] + 2*rPoint[1]*rPoint[2] + 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 0, 2 ) = (+0.125) * (1 - rPoint[0]) * (1 - rPoint[1]) * (1 + rPoint[0] + rPoint[1] + 2*rPoint[2] + rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] + 2*rPoint[0]*rPoint[1]*rPoint[2]) ;

rResult( 1, 0 ) = (-0.0625) * (1 - rPoint[1]) * (1 - rPoint[2]) * (1 - 6*rPoint[0] + rPoint[1] - 4*rPoint[0]*rPoint[1] + rPoint[2] - 2*rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] - 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 1, 1 ) = (+0.0625) * (1 + rPoint[0]) * (1 - rPoint[2]) * (1 - rPoint[0] + 6*rPoint[1] - 4*rPoint[0]*rPoint[1] + rPoint[2] + rPoint[0]*rPoint[2] + 2*rPoint[1]*rPoint[2] - 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 1, 2 ) = (+0.125) * (1 + rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[0] + rPoint[1] + 2*rPoint[2] - rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] - 2*rPoint[0]*rPoint[1]*rPoint[2]) ;

rResult( 2, 0 ) = (-0.0625) * (1 + rPoint[1]) * (1 - rPoint[2]) * (1 - 6*rPoint[0] - rPoint[1] + 4*rPoint[0]*rPoint[1] + rPoint[2] - 2*rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] + 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 2, 1 ) = (-0.0625) * (1 + rPoint[0]) * (1 - rPoint[2]) * (1 - rPoint[0] - 6*rPoint[1] + 4*rPoint[0]*rPoint[1] + rPoint[2] + rPoint[0]*rPoint[2] - 2*rPoint[1]*rPoint[2] + 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 2, 2 ) = (+0.125) * (1 + rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[0] - rPoint[1] + 2*rPoint[2] - rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] + 2*rPoint[0]*rPoint[1]*rPoint[2]) ;

rResult( 3, 0 ) = (+0.0625) * (1 + rPoint[1]) * (1 - rPoint[2]) * (1 + 6*rPoint[0] - rPoint[1] - 4*rPoint[0]*rPoint[1] + rPoint[2] + 2*rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] - 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 3, 1 ) = (-0.0625) * (1 - rPoint[0]) * (1 - rPoint[2]) * (1 + rPoint[0] - 6*rPoint[1] - 4*rPoint[0]*rPoint[1] + rPoint[2] - rPoint[0]*rPoint[2] - 2*rPoint[1]*rPoint[2] - 4*rPoint[0]*rPoint[1]*rPoint[2]) ;
rResult( 3, 2 ) = (+0.125) * (1 - rPoint[0]) * (1 + rPoint[1]) * (1 + rPoint[0] - rPoint[1] + 2*rPoint[2] + rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] - 2*rPoint[0]*rPoint[1]*rPoint[2]) ;

rResult( 4, 0 ) = 0.00 ;
rResult( 4, 1 ) = 0.00 ;
rResult( 4, 2 ) = (0.5) + rPoint[2] ;

rResult( 5, 0 ) = (-0.25) * (rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[2]) * (2 + rPoint[1] + rPoint[1]*rPoint[2]) ;
rResult( 5, 1 ) = (-0.125) * (1 - std::pow(rPoint[0],2.0)) * (1 - rPoint[2]) * (1 + 2*rPoint[1] - rPoint[2] + 2*rPoint[1]*rPoint[2]) ;
rResult( 5, 2 ) = (-0.25) * (1 - std::pow(rPoint[0],2.0)) * (1 - rPoint[1]) * (1 + rPoint[1]*rPoint[2]) ;

rResult( 6, 0 ) = (0.125) * (1 - std::pow(rPoint[1],2.0)) * (1 - rPoint[2]) * (1 - 2*rPoint[0] - rPoint[2] - 2*rPoint[0]*rPoint[2]) ;
rResult( 6, 1 ) = (-0.25) * (1 + rPoint[0]) * (rPoint[1]) *  (1 - rPoint[2]) * (2 - rPoint[0] - rPoint[0]*rPoint[2]) ;
rResult( 6, 2 ) = (-0.25) * (1 + rPoint[0]) * (1 - std::pow(rPoint[1],2.0)) *  (1 - rPoint[0]*rPoint[2]) ;

rResult( 7, 0 ) = (-0.25) * (rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[2]) * (2 - rPoint[1] - rPoint[1]*rPoint[2]) ;
rResult( 7, 1 ) = (+0.125) * (1 - std::pow(rPoint[0],2.0)) * (1 - rPoint[2]) * (1 - 2*rPoint[1] - rPoint[2] - 2*rPoint[1]*rPoint[2]) ;
rResult( 7, 2 ) = (-0.25) * (1 - std::pow(rPoint[0],2.0)) * (1 + rPoint[1]) * (1 - rPoint[1]*rPoint[2]) ;

rResult( 8, 0 ) = (-0.125) * (1 - std::pow(rPoint[1],2.0)) * (1 - rPoint[2]) * (1 + 2*rPoint[0] - rPoint[2] + 2*rPoint[0]*rPoint[2]) ;
rResult( 8, 1 ) = (-0.25) * (1 - rPoint[0]) * (rPoint[1]) *  (1 - rPoint[2]) * (2 + rPoint[0] + rPoint[0]*rPoint[2]) ;
rResult( 8, 2 ) = (-0.25) * (1 - rPoint[0]) * (1 - std::pow(rPoint[1],2.0)) *  (1 + rPoint[0]*rPoint[2]) ;

rResult( 9, 0 ) = (-0.25) * (1 - rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 9, 1 ) = (-0.25) * (1 - rPoint[0]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 9, 2 ) = (-0.5) * (1 - rPoint[0]) * (1 - rPoint[1]) * (rPoint[2]) ;

rResult( 10, 0 ) = (+0.25) * (1 - rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 10, 1 ) = (-0.25) * (1 + rPoint[0]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 10, 2 ) = (-0.5) * (1 + rPoint[0]) * (1 - rPoint[1]) * (rPoint[2]) ;

rResult( 11, 0 ) = (+0.25) * (1 + rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 11, 1 ) = (+0.25) * (1 + rPoint[0]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 11, 2 ) = (-0.5) * (1 + rPoint[0]) * (1 + rPoint[1]) * (rPoint[2]) ;

rResult( 12, 0 ) = (-0.25) * (1 + rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 12, 1 ) = (+0.25) * (1 - rPoint[0]) * (1 - std::pow(rPoint[2],2.0)) ;
rResult( 12, 2 ) = (-0.5) * (1 - rPoint[0]) * (1 + rPoint[1]) * (rPoint[2]) ;

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
return "3 dimensional pyramid with 13 nodes in 3D space";
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

Pyramid3D13() : BaseType( PointsArrayType(), &msGeometryData ) {}


static double ShapeFunctionValueImpl(
IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint)
{
switch ( ShapeFunctionIndex )
{
case 0:
return( (-0.0625) * (1 - rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[2]) * (4 + 3*rPoint[0] + 3*rPoint[1] + 2*rPoint[0]*rPoint[1] + 2*rPoint[2] + rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] + 2*rPoint[0]*rPoint[1]*rPoint[2]) );
case 1:
return( (-0.0625) * (1 + rPoint[0]) * (1 - rPoint[1]) * (1 - rPoint[2]) * (4 - 3*rPoint[0] + 3*rPoint[1] - 2*rPoint[0]*rPoint[1] + 2*rPoint[2] - rPoint[0]*rPoint[2] + rPoint[1]*rPoint[2] - 2*rPoint[0]*rPoint[1]*rPoint[2]) );
case 2:
return( (-0.0625) * (1 + rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[2]) * (4 - 3*rPoint[0] - 3*rPoint[1] + 2*rPoint[0]*rPoint[1] + 2*rPoint[2] - rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] + 2*rPoint[0]*rPoint[1]*rPoint[2]) );
case 3:
return( (-0.0625) * (1 - rPoint[0]) * (1 + rPoint[1]) * (1 - rPoint[2]) * (4 + 3*rPoint[0] - 3*rPoint[1] - 2*rPoint[0]*rPoint[1] + 2*rPoint[2] + rPoint[0]*rPoint[2] - rPoint[1]*rPoint[2] - 2*rPoint[0]*rPoint[1]*rPoint[2]) );
case 4:
return( (0.5) * (rPoint[2]) * (1 + rPoint[2]) );
case 5:
return( (0.125) * (1 - std::pow(rPoint[0],2.0)) * (1 - rPoint[1]) * (1 - rPoint[2]) * (2 + rPoint[1] + rPoint[1]*rPoint[2]) );
case 6:
return( (0.125) * (1 + rPoint[0]) * (1 - std::pow(rPoint[1],2.0)) * (1 - rPoint[2]) * (2 - rPoint[0] - rPoint[0]*rPoint[2]) );
case 7:
return( (0.125) * (1 - std::pow(rPoint[0],2.0)) * (1 + rPoint[1]) * (1 - rPoint[2]) * (2 - rPoint[1] - rPoint[1]*rPoint[2]) );
case 8:
return( (0.125) * (1 - rPoint[0]) * (1 - std::pow(rPoint[1],2.0)) * (1 - rPoint[2]) * (2 + rPoint[0] + rPoint[0]*rPoint[2]) );
case 9:
return( (0.25) * (1 - rPoint[0]) * (1 - rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) );
case 10:
return( (0.25) * (1 + rPoint[0]) * (1 - rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) );
case 11:
return( (0.25) * (1 + rPoint[0]) * (1 + rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) );
case 12:
return( (0.25) * (1 - rPoint[0]) * (1 + rPoint[1]) * (1 - std::pow(rPoint[2],2.0)) );
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
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_1),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_2),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_3),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_4),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients =
{
{
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_1),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_2),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_3),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_4),
Pyramid3D13<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Pyramid3D13;


}; 





template<class TPointType>
inline std::istream& operator >> (std::istream& rIStream,
Pyramid3D13<TPointType>& rThis);

template<class TPointType>
inline std::ostream& operator << (std::ostream& rOStream,
const Pyramid3D13<TPointType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

template<class TPointType> const
GeometryData Pyramid3D13<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Pyramid3D13<TPointType>::AllIntegrationPoints(),
Pyramid3D13<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Pyramid3D13<TPointType>::msGeometryDimension(3, 3);

}  
