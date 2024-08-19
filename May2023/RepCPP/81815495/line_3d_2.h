
#pragma once



#include "geometries/geometry.h"
#include "integration/line_gauss_legendre_integration_points.h"
#include "integration/line_collocation_integration_points.h"
#include "utilities/intersection_utilities.h"
#include "utilities/geometry_utilities.h"
namespace Kratos
{







template<class TPointType>

class Line3D2 : public Geometry<TPointType>
{

public:

typedef Geometry<TPointType> BaseType;
using Geometry<TPointType>::ShapeFunctionsValues;

KRATOS_CLASS_POINTER_DEFINITION( Line3D2 );

typedef Line3D2<TPointType> EdgeType;


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



Line3D2( typename PointType::Pointer pFirstPoint, typename PointType::Pointer pSecondPoint )
: BaseType( PointsArrayType(), &msGeometryData )
{
BaseType::Points().push_back( pFirstPoint );
BaseType::Points().push_back( pSecondPoint );
}

explicit Line3D2( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( BaseType::PointsNumber() != 2 )
KRATOS_ERROR << "Invalid points number. Expected 2, given " << BaseType::PointsNumber() << std::endl;
}

explicit Line3D2(
const IndexType GeometryId,
const PointsArrayType& ThisPoints
) : BaseType(GeometryId, ThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 2 ) << "Invalid points number. Expected 2, given " << this->PointsNumber() << std::endl;
}

explicit Line3D2(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 2) << "Invalid points number. Expected 2, given " << this->PointsNumber() << std::endl;
}


Line3D2( Line3D2 const& rOther )
: BaseType( rOther )
{
}



template<class TOtherPointType> explicit Line3D2( Line3D2<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Line3D2() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Linear;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Line3D2;
}



Line3D2& operator=( const Line3D2& rOther )
{
BaseType::operator=( rOther );

return *this;
}


template<class TOtherPointType>
Line3D2& operator=( Line3D2<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Line3D2( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Line3D2( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Vector& LumpingFactors(
Vector& rResult,
const typename BaseType::LumpingMethods LumpingMethod = BaseType::LumpingMethods::ROW_SUM
)  const override
{
if(rResult.size() != 2)
rResult.resize( 2, false );
rResult[0] = 0.5;
rResult[1] = 0.5;
return rResult;
}



double Length() const override
{
const TPointType& point0 = BaseType::GetPoint(0);
const TPointType& point1 = BaseType::GetPoint(1);
const double lx = point0.X() - point1.X();
const double ly = point0.Y() - point1.Y();
const double lz = point0.Z() - point1.Z();

const double length = lx * lx + ly * ly + lz * lz;

return std::sqrt( length );
}


double Area() const override
{
return Length();
}



double DomainSize() const override
{
return Length();
}




bool HasIntersection(const BaseType& rThisGeometry) const override
{
const BaseType& r_geom = *this;
if (rThisGeometry.LocalSpaceDimension() > r_geom.LocalSpaceDimension()) {
return rThisGeometry.HasIntersection(r_geom);
}
Point intersection_point;
return IntersectionUtilities::ComputeLineLineIntersection(r_geom, rThisGeometry[0], rThisGeometry[1], intersection_point);
}



double CalculateDistance(
const CoordinatesArrayType& rPointGlobalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
const Point point(rPointGlobalCoordinates);
return GeometryUtils::PointDistanceToLineSegment3D(this->GetPoint(0), this->GetPoint(1), point);
}



JacobiansType& Jacobian( JacobiansType& rResult, IntegrationMethod ThisMethod ) const override
{
Matrix jacobian( 3, 1 );
jacobian( 0, 0 ) = ( this->GetPoint( 1 ).X() - this->GetPoint( 0 ).X() ) * 0.5; 
jacobian( 1, 0 ) = ( this->GetPoint( 1 ).Y() - this->GetPoint( 0 ).Y() ) * 0.5;
jacobian( 2, 0 ) = ( this->GetPoint( 1 ).Z() - this->GetPoint( 0 ).Z() ) * 0.5;

if ( rResult.size() != BaseType::IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( BaseType::IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

std::fill( rResult.begin(), rResult.end(), jacobian );

return rResult;
}


JacobiansType& Jacobian( JacobiansType& rResult, IntegrationMethod ThisMethod, Matrix & DeltaPosition ) const override
{
Matrix jacobian( 3, 1 );
jacobian( 0, 0 ) = ( (this->GetPoint( 1 ).X() - DeltaPosition(1,0)) - (this->GetPoint( 0 ).X() - DeltaPosition(0,0)) ) * 0.5; 
jacobian( 1, 0 ) = ( (this->GetPoint( 1 ).Y() - DeltaPosition(1,1)) - (this->GetPoint( 0 ).Y() - DeltaPosition(0,1)) ) * 0.5;
jacobian( 2, 0 ) = ( (this->GetPoint( 1 ).Z() - DeltaPosition(1,2)) - (this->GetPoint( 0 ).Z() - DeltaPosition(0,2)) ) * 0.5;

if ( rResult.size() != BaseType::IntegrationPointsNumber( ThisMethod ) )
{
JacobiansType temp( BaseType::IntegrationPointsNumber( ThisMethod ) );
rResult.swap( temp );
}

std::fill( rResult.begin(), rResult.end(), jacobian );

return rResult;
}


Matrix& Jacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const override
{
rResult.resize( 3, 1, false );
rResult( 0, 0 ) = ( this->GetPoint( 1 ).X() - this->GetPoint( 0 ).X() ) * 0.5;
rResult( 1, 0 ) = ( this->GetPoint( 1 ).Y() - this->GetPoint( 0 ).Y() ) * 0.5;
rResult( 2, 0 ) = ( this->GetPoint( 1 ).Z() - this->GetPoint( 0 ).Z() ) * 0.5;
return rResult;
}


Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
rResult.resize( 3, 1, false );
rResult( 0, 0 ) = ( this->GetPoint( 1 ).X() - this->GetPoint( 0 ).X() ) * 0.5;
rResult( 1, 0 ) = ( this->GetPoint( 1 ).Y() - this->GetPoint( 0 ).Y() ) * 0.5;
rResult( 2, 0 ) = ( this->GetPoint( 1 ).Z() - this->GetPoint( 0 ).Z() ) * 0.5;
return rResult;
}


Vector& DeterminantOfJacobian( Vector& rResult, IntegrationMethod ThisMethod ) const override
{
const unsigned int integration_points_number = msGeometryData.IntegrationPointsNumber( ThisMethod );
if(rResult.size() != integration_points_number)
{
rResult.resize(integration_points_number,false);
}

const double detJ = 0.5*(this->Length());

for ( unsigned int pnt = 0; pnt < integration_points_number; pnt++ )
{
rResult[pnt] = detJ;
}
return rResult;
}


double DeterminantOfJacobian( IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const override
{
return 0.5*(this->Length());
}


double DeterminantOfJacobian( const CoordinatesArrayType& rPoint ) const override
{
return 0.5*(this->Length());
}



JacobiansType& InverseOfJacobian( JacobiansType& rResult, IntegrationMethod ThisMethod ) const override
{
rResult[0] = ZeroMatrix( 1, 1 );
rResult[0]( 0, 0 ) = 2.0 * MathUtils<double>::Norm3(( this->GetPoint( 1 ) ) - ( this->GetPoint( 0 ) ) );
return rResult;
}


Matrix& InverseOfJacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const override
{
rResult = ZeroMatrix( 1, 1 );
rResult( 0, 0 ) = 2.0 * MathUtils<double>::Norm3(( this->GetPoint( 1 ) ) - ( this->GetPoint( 0 ) ) );
return( rResult );
}


Matrix& InverseOfJacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
rResult = ZeroMatrix( 1, 1 );
rResult( 0, 0 ) = 2.0 * MathUtils<double>::Norm3(( this->GetPoint( 1 ) ) - ( this->GetPoint( 0 ) ) );
return( rResult );
}



SizeType EdgesNumber() const override
{
return 1;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
edges.push_back( Kratos::make_shared<EdgeType>( this->pGetPoint( 0 ), this->pGetPoint( 1 ) ) );
return edges;
}



SizeType FacesNumber() const override
{
return 0;
}



Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 2) {
rResult.resize(2, false);
}

rResult[0] =  0.5 * ( 1.0 - rCoordinates[0]);
rResult[1] =  0.5 * ( 1.0 + rCoordinates[0]);

return rResult;
}


double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
switch ( ShapeFunctionIndex )
{
case 0:
return( 0.5*( 1.0 - rPoint[0] ) );

case 1:
return( 0.5*( 1.0 + rPoint[0] ) );

default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}

Matrix& ShapeFunctionsLocalGradients( Matrix& rResult,
const CoordinatesArrayType& rPoint ) const override
{
rResult = ZeroMatrix( 2, 1 );
rResult( 0, 0 ) = -0.5;
rResult( 1, 0 ) =  0.5;
return( rResult );
}


array_1d<double, 3> Normal(const CoordinatesArrayType& rPointLocalCoordinates) const override
{
KRATOS_ERROR << "ERROR: Line3D2 can not define a normal. Please, define the normal in your implementation" << std::endl;
return ZeroVector(3);
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
PointLocalCoordinates( rResult, rPoint );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) ) {
return true;
}

return false;
}


bool HasIntersection(const Point& rLowPoint, const Point& rHighPoint) const override
{
return IntersectionUtilities::ComputeLineBoxIntersection(
rLowPoint, rHighPoint, this->GetPoint(0), this->GetPoint(1));
}


CoordinatesArrayType& PointLocalCoordinates(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint
) const override
{
rResult.clear();

const TPointType& r_first_point  = BaseType::GetPoint(0);
const TPointType& r_second_point = BaseType::GetPoint(1);

const double tolerance = 1e-14; 

const double length = Length();

const double length_1 = std::sqrt( std::pow(rPoint[0] - r_first_point[0], 2)
+ std::pow(rPoint[1] - r_first_point[1], 2) + std::pow(rPoint[2] - r_first_point[2], 2));

const double length_2 = std::sqrt( std::pow(rPoint[0] - r_second_point[0], 2)
+ std::pow(rPoint[1] - r_second_point[1], 2) + std::pow(rPoint[2] - r_second_point[2], 2));

if (length_1 <= (length + tolerance) && length_2 <= (length + tolerance)) {
rResult[0] = 2.0 * length_1/(length + tolerance) - 1.0;
} else if (length_1 > (length + tolerance)) {
rResult[0] = 2.0 * length_1/(length + tolerance) - 1.0; 
} else if (length_2 > (length + tolerance)) {
rResult[0] = 1.0 - 2.0 * length_2/(length + tolerance);
} else {
rResult[0] = 2.0; 
}

return rResult ;
}



std::string Info() const override
{
return "1 dimensional line with 2 nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "1 dimensional line with 2 nodes in 3D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
Jacobian( jacobian, PointType() );
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

Line3D2(): BaseType( PointsArrayType(), &msGeometryData ) {}





static Matrix CalculateShapeFunctionsIntegrationPointsValues( typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
int integration_points_number = IntegrationPoints.size();
Matrix N( integration_points_number, 2 );

for ( int it_gp = 0; it_gp < integration_points_number; it_gp++ )
{
double e = IntegrationPoints[it_gp].X();
N( it_gp, 0 ) = 0.5 * ( 1 - e );
N( it_gp, 1 ) = 0.5 * ( 1 + e );
}

return N;
}

static ShapeFunctionsGradientsType CalculateShapeFunctionsIntegrationPointsLocalGradients( typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
ShapeFunctionsGradientsType DN_De( IntegrationPoints.size() );

for ( unsigned int it_gp = 0; it_gp < IntegrationPoints.size(); it_gp++ )
{
Matrix aux_mat = ZeroMatrix(2,1);
aux_mat(0,0) = -0.5;
aux_mat(1,0) =  0.5;
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
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients = {{
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line3D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )

}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Line3D2;



}; 






template<class TPointType>
inline std::istream& operator >> ( std::istream& rIStream,
Line3D2<TPointType>& rThis );

template<class TPointType>
inline std::ostream& operator << ( std::ostream& rOStream,
const Line3D2<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}



template<class TPointType>
const GeometryData Line3D2<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
Line3D2<TPointType>::AllIntegrationPoints(),
Line3D2<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients() );

template<class TPointType>
const GeometryDimension Line3D2<TPointType>::msGeometryDimension(3, 1);

}  