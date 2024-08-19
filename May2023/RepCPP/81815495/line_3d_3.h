
#pragma once



#include "geometries/geometry.h"
#include "integration/line_gauss_legendre_integration_points.h"
#include "integration/line_collocation_integration_points.h"
#include "utilities/integration_utilities.h"

namespace Kratos
{







template<class TPointType>

class Line3D3 : public Geometry<TPointType>
{
public:

typedef Geometry<TPointType> BaseType;

KRATOS_CLASS_POINTER_DEFINITION( Line3D3 );


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef typename BaseType::GeometriesArrayType GeometriesArrayType;


typedef TPointType PointType;


typedef typename BaseType::IndexType IndexType;



typedef typename BaseType::SizeType SizeType;


typedef  typename BaseType::PointsArrayType PointsArrayType;


typedef typename BaseType::IntegrationPointType IntegrationPointType;


typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


typedef typename BaseType::IntegrationPointsContainerType IntegrationPointsContainerType;


typedef typename BaseType::ShapeFunctionsValuesContainerType ShapeFunctionsValuesContainerType;


typedef typename BaseType::ShapeFunctionsLocalGradientsContainerType ShapeFunctionsLocalGradientsContainerType;


typedef typename BaseType::JacobiansType JacobiansType;


typedef typename BaseType::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef typename BaseType::NormalType NormalType;


typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;


Line3D3( typename PointType::Pointer pFirstPoint, typename PointType::Pointer pSecondPoint,
typename PointType::Pointer pThirdPoint )
: BaseType( PointsArrayType(), &msGeometryData )
{
BaseType::Points().push_back( pFirstPoint );
BaseType::Points().push_back( pSecondPoint );
BaseType::Points().push_back( pThirdPoint );
}

Line3D3( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( BaseType::PointsNumber() != 3 )
KRATOS_ERROR << "Invalid points number. Expected 3, given " << BaseType::PointsNumber() << std::endl;
}

explicit Line3D3(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 3 ) << "Invalid points number. Expected 3, given " << this->PointsNumber() << std::endl;
}

explicit Line3D3(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 3) << "Invalid points number. Expected 3, given " << this->PointsNumber() << std::endl;
}


Line3D3( Line3D3 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Line3D3( Line3D3<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Line3D3() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Linear;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Line3D3;
}



Line3D3& operator=( const Line3D3& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Line3D3& operator=( Line3D3<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );
return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Line3D3(NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Line3D3( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Vector& LumpingFactors(
Vector& rResult,
const typename BaseType::LumpingMethods LumpingMethod = BaseType::LumpingMethods::ROW_SUM
)  const override
{
if(rResult.size() != 3)
rResult.resize( 3, false );
rResult[0] = 1.0/6.0;
rResult[2] = 2.0/3.0;
rResult[1] = 1.0/6.0;
return rResult;
}



double Length() const override
{
const IntegrationMethod integration_method = IntegrationUtilities::GetIntegrationMethodForExactMassMatrixEvaluation(*this);
return IntegrationUtilities::ComputeDomainSize(*this, integration_method);
}


double Area() const override
{
return Length();
}



double DomainSize() const override
{
return Length();
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) )
return true;

return false;
}


Vector& DeterminantOfJacobian( Vector& rResult, IntegrationMethod ThisMethod ) const override
{
const std::size_t number_of_integration_points = this->IntegrationPointsNumber( ThisMethod );
if( rResult.size() != number_of_integration_points)
rResult.resize( number_of_integration_points, false );

Matrix J(3, 1);
for (std::size_t pnt = 0; pnt < number_of_integration_points; ++pnt) {
this->Jacobian( J, pnt, ThisMethod);
rResult[pnt] = std::sqrt(std::pow(J(0,0), 2) + std::pow(J(1,0), 2) + std::pow(J(2,0), 2));
}
return rResult;
}


double DeterminantOfJacobian(
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod ) const override
{
Matrix J(3, 1);
this->Jacobian( J, IntegrationPointIndex, ThisMethod);
return std::sqrt(std::pow(J(0,0), 2) + std::pow(J(1,0), 2) + std::pow(J(2,0), 2));
}


double DeterminantOfJacobian( const CoordinatesArrayType& rPoint ) const override
{
Matrix J(3, 1);
this->Jacobian( J, rPoint);
return std::sqrt(std::pow(J(0,0), 2) + std::pow(J(1,0), 2) + std::pow(J(2,0), 2));
}



SizeType EdgesNumber() const override
{
return 2;
}

SizeType FacesNumber() const override
{
return 0;
}



Vector& ShapeFunctionsValues(
Vector& rResult,
const CoordinatesArrayType& rCoordinates
) const override
{
if(rResult.size() != 3) {
rResult.resize(3, false);
}

rResult[0] = 0.5 * (rCoordinates[0] - 1.0) * rCoordinates[0];
rResult[1] = 0.5 * (rCoordinates[0] + 1.0) * rCoordinates[0];
rResult[2] = 1.0 - rCoordinates[0] * rCoordinates[0];

return rResult;
}


double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return( 0.5*( rPoint[0] - 1.0 )*rPoint[0] );
case 1:
return( 0.5*( rPoint[0] + 1.0 )*rPoint[0] );
case 2:
return( 1.0 -rPoint[0]*rPoint[0] );

default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
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
if(rResult.size1() != 3 || rResult.size2() != 1) {
rResult.resize( 3, 1, false );
}

noalias( rResult ) = ZeroMatrix( 3, 1 );
rResult( 0, 0 ) =  rPoint[0] - 0.5;
rResult( 1, 0 ) =  rPoint[0] + 0.5;
rResult( 2, 0 ) = -rPoint[0] * 2.0;
return( rResult );
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
if(rResult.size1() != 3 || rResult.size2() != 1) {
rResult.resize( 3, 1, false );
}
noalias( rResult ) = ZeroMatrix( 3, 1 );
rResult( 0, 0 ) = -1.0;
rResult( 1, 0 ) =  1.0;
rResult( 2, 0 ) =  0.0;
return rResult;
}


virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, CoordinatesArrayType& rPoint )
{
if(rResult.size1() != 3 || rResult.size2() != 1) {
rResult.resize( 3, 1, false );
}
noalias( rResult ) = ZeroMatrix( 3, 1 );

rResult( 0, 0 ) =  rPoint[0] - 0.5;
rResult( 1, 0 ) =  rPoint[0] + 0.5;
rResult( 2, 0 ) = -rPoint[0] * 2.0;
return rResult;
}



std::string Info() const override
{
return "1 dimensional line with 3 nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "1 dimensional line with 3 nodes in 3D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
rOStream << "    Jacobian\t : " << jacobian;
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

Line3D3(): BaseType( PointsArrayType(), &msGeometryData ) {}



static Matrix CalculateShapeFunctionsIntegrationPointsValues( typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
int integration_points_number = IntegrationPoints.size();
Matrix N( integration_points_number, 3 );

for ( int it_gp = 0; it_gp < integration_points_number; it_gp++ )
{
double e = IntegrationPoints[it_gp].X();
N( it_gp, 0 ) = 0.5 * ( e - 1 ) * e;
N( it_gp, 2 ) = 1.0 - e * e;
N( it_gp, 1 ) = 0.5 * ( 1 + e ) * e;
}

return N;
}

static ShapeFunctionsGradientsType CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
ShapeFunctionsGradientsType DN_De( IntegrationPoints.size() );
std::fill( DN_De.begin(), DN_De.end(), Matrix( 3, 1 ) );

for ( unsigned int it_gp = 0; it_gp < IntegrationPoints.size(); it_gp++ )
{
Matrix aux_mat = ZeroMatrix(3,1);
const double e = IntegrationPoints[it_gp].X();
aux_mat(0,0) = e - 0.5;
aux_mat(2,0) = -2.0 * e;
aux_mat(1,0) = e + 0.5;
DN_De[it_gp] = aux_mat;
}

return DN_De;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points = {{
Quadrature<LineGaussLegendreIntegrationPoints1, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineGaussLegendreIntegrationPoints2, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineGaussLegendreIntegrationPoints3, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineGaussLegendreIntegrationPoints4, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineGaussLegendreIntegrationPoints5, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineCollocationIntegrationPoints1, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineCollocationIntegrationPoints2, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineCollocationIntegrationPoints3, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineCollocationIntegrationPoints4, 1, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature<LineCollocationIntegrationPoints5, 1, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values = {{
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients = {{
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line3D3<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Line3D3;



}; 






template<class TPointType>
inline std::istream& operator >> ( std::istream& rIStream,
Line3D3<TPointType>& rThis );

template<class TPointType>
inline std::ostream& operator << ( std::ostream& rOStream,
const Line3D3<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}



template<class TPointType>
const GeometryData Line3D3<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Line3D3<TPointType>::AllIntegrationPoints(),
Line3D3<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients() );

template<class TPointType>
const GeometryDimension Line3D3<TPointType>::msGeometryDimension(3, 1);

}  
