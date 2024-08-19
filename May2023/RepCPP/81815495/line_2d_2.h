
#pragma once



#include "geometries/geometry.h"
#include "integration/line_gauss_legendre_integration_points.h"
#include "integration/line_collocation_integration_points.h"
#include "utilities/geometrical_projection_utilities.h"

namespace Kratos
{







template<class TPointType>

class Line2D2 : public Geometry<TPointType>
{
public:

typedef Geometry<TPointType> BaseType;
using Geometry<TPointType>::ShapeFunctionsValues;

KRATOS_CLASS_POINTER_DEFINITION( Line2D2 );

typedef Line2D2<TPointType> EdgeType;


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





Line2D2( typename PointType::Pointer pFirstPoint, typename PointType::Pointer pSecondPoint )
: BaseType( PointsArrayType(), &msGeometryData )
{
BaseType::Points().push_back( pFirstPoint );
BaseType::Points().push_back( pSecondPoint );
}


explicit Line2D2( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( BaseType::PointsNumber() != 2 )
KRATOS_ERROR << "Invalid points number. Expected 2, given " << BaseType::PointsNumber() << std::endl;
}

explicit Line2D2(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 2 ) << "Invalid points number. Expected 2, given " << this->PointsNumber() << std::endl;
}

explicit Line2D2(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 2) << "Invalid points number. Expected 2, given " << this->PointsNumber() << std::endl;
}


Line2D2( Line2D2 const& rOther )
: BaseType( rOther )
{
}



template<class TOtherPointType> explicit Line2D2( Line2D2<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Line2D2() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Linear;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Line2D2;
}



Line2D2& operator=( const Line2D2& rOther )
{
BaseType::operator=( rOther );

return *this;
}


template<class TOtherPointType>
Line2D2& operator=( Line2D2<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}



typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Line2D2( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Line2D2( NewGeometryId, rGeometry.Points() ) );
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
const TPointType& FirstPoint  = BaseType::GetPoint(0);
const TPointType& SecondPoint = BaseType::GetPoint(1);
const double lx = FirstPoint.X() - SecondPoint.X();
const double ly = FirstPoint.Y() - SecondPoint.Y();

const double length = lx * lx + ly * ly;

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



JacobiansType& Jacobian( JacobiansType& rResult, IntegrationMethod ThisMethod ) const override
{
Matrix jacobian( 2, 1 );
jacobian( 0, 0 ) = ( BaseType::GetPoint( 1 ).X() - BaseType::GetPoint( 0 ).X() ) * 0.5; 
jacobian( 1, 0 ) = ( BaseType::GetPoint( 1 ).Y() - BaseType::GetPoint( 0 ).Y() ) * 0.5;

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
Matrix jacobian( 2, 1 );
jacobian( 0, 0 ) = ( (BaseType::GetPoint( 1 ).X() - DeltaPosition(1,0)) - (BaseType::GetPoint( 0 ).X() - DeltaPosition(0,0)) ) * 0.5; 
jacobian( 1, 0 ) = ( (BaseType::GetPoint( 1 ).Y() - DeltaPosition(1,1)) - (BaseType::GetPoint( 0 ).Y() - DeltaPosition(0,1)) ) * 0.5;

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
rResult.resize( 2, 1, false );
rResult( 0, 0 ) = ( BaseType::GetPoint( 1 ).X() - BaseType::GetPoint( 0 ).X() ) * 0.5;
rResult( 1, 0 ) = ( BaseType::GetPoint( 1 ).Y() - BaseType::GetPoint( 0 ).Y() ) * 0.5;
return rResult;
}


Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rPoint ) const override
{
rResult.resize( 2, 1, false );
rResult( 0, 0 ) = ( BaseType::GetPoint( 1 ).X() - BaseType::GetPoint( 0 ).X() ) * 0.5;
rResult( 1, 0 ) = ( BaseType::GetPoint( 1 ).Y() - BaseType::GetPoint( 0 ).Y() ) * 0.5;
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
return rResult;
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

void NumberNodesInFaces (DenseVector<unsigned int>& NumberNodesInFaces) const override
{
if(NumberNodesInFaces.size() != 2 )
NumberNodesInFaces.resize(2,false);

NumberNodesInFaces[0]=1;
NumberNodesInFaces[1]=1;

}

void NodesInFaces (DenseMatrix<unsigned int>& NodesInFaces) const override
{
if(NodesInFaces.size1() != 2 || NodesInFaces.size2() != 2)
NodesInFaces.resize(2,2,false);

NodesInFaces(0,0)=0;
NodesInFaces(1,0)=1;

NodesInFaces(0,1)=1;
NodesInFaces(1,1)=0;
}



Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 2)
{
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
return( 0.5 * ( 1.0 - rPoint[0] ) );
case 1:
return( 0.5 * ( 1.0 + rPoint[0] ) );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}



std::string Info() const override
{
return "1 dimensional line in 2D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "1 dimensional line in 2D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
Jacobian( jacobian, PointType() );
rOStream << "    Jacobian\t : " << jacobian;
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
if(rResult.size1() != 2 || rResult.size2() != 1)
{
rResult.resize( 2, 1, false );
}
noalias( rResult ) = ZeroMatrix( 2, 1 );
rResult( 0, 0 ) = - 0.5;
rResult( 1, 0 ) =   0.5;

return( rResult );
}


array_1d<double, 3> Normal(const CoordinatesArrayType& rPointLocalCoordinates) const override
{
array_1d<double,3> normal;

const TPointType& first_point  = BaseType::GetPoint(0);
const TPointType& second_point = BaseType::GetPoint(1);

normal[0] = second_point[1] -  first_point[1];
normal[1] =  first_point[0] - second_point[0];
normal[2] = 0.0;

return normal;
}


Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
if(rResult.size1() != 2 || rResult.size2() != 1)
{
rResult.resize( 2, 1, false );
}
noalias( rResult ) = ZeroMatrix( 2, 1 );
rResult( 0, 0 ) = -1.0;
rResult( 1, 0 ) =  1.0;
return rResult;
}


virtual Matrix& ShapeFunctionsGradients( Matrix& rResult, CoordinatesArrayType& rPoint )
{
if(rResult.size1() != 2 || rResult.size2() != 1)
{
rResult.resize( 2, 1, false );
}
noalias( rResult ) = ZeroMatrix( 2, 1 );

rResult( 0, 0 ) = - 0.5;
rResult( 1, 0 ) =   0.5;
return rResult;
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
const Point point_to_project(rPoint);
Point point_projected;
const double distance = GeometricalProjectionUtilities::FastProjectOnLine2D(*this, point_to_project, point_projected);

if (std::abs(distance) > std::numeric_limits<double>::epsilon()) {
if (std::abs(distance) > 1.0e-6 * Length()) {
KRATOS_WARNING_FIRST_N("Line2D2", 10) << "The point of coordinates X: " << rPoint[0] << "\tY: " << rPoint[1] << " it is in a distance: " << std::abs(distance) << std::endl;
return false;
}
}

PointLocalCoordinates( rResult, point_projected );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) ) {
return true;
}

return false;
}


bool HasIntersection(const BaseType& rOtherGeometry) const override
{
const double tolerance = std::numeric_limits<double>::epsilon();
const TPointType& first_point  = BaseType::GetPoint(0); 
const TPointType& second_point = BaseType::GetPoint(1); 

const TPointType& first_point_other  = *rOtherGeometry(0); 
const TPointType& second_point_other = *rOtherGeometry(1); 

const double numerator   = ( (first_point[0]-first_point_other[0])*(first_point_other[1] - second_point_other[1]) - (first_point[1]-first_point_other[1])*(first_point_other[0]-second_point_other[0]) );
const double denominator = ( (first_point[0]-second_point[0])*(first_point_other[1] - second_point_other[1]) - (first_point[1]-second_point[1])*(first_point_other[0]-second_point_other[0]) );
if (std::abs(denominator) < tolerance) 
return false;
const double t = numerator  /  denominator;

return (0.0-tolerance<=t) && (t<=1.0+tolerance);
}


bool HasIntersection(const Point& rLowPoint, const Point& rHighPoint) const override
{
const double tolerance = std::numeric_limits<double>::epsilon();
const TPointType& first_point  = BaseType::GetPoint(0);
const TPointType& second_point = BaseType::GetPoint(1);

if (    
( (first_point[0] >= rLowPoint[0] && first_point[0] <= rHighPoint[0])
&& (first_point[1] >= rLowPoint[1] && first_point[1] <= rHighPoint[1]) ) 
||
( (second_point[0] >= rLowPoint[0] && second_point[0] <= rHighPoint[0])
&& (second_point[1] >= rLowPoint[1] && second_point[1] <= rHighPoint[1]) ) 
)
return true;

const double high_x = rHighPoint[0];
const double high_y = rHighPoint[1];
const double low_x = rLowPoint[0];
const double low_y = rLowPoint[1];

const double denominator = ( second_point[0] - first_point[0] );
const double numerator = (second_point[1] - first_point[1]);
const double slope = std::abs(denominator) > tolerance ? std::abs(numerator) > tolerance ? numerator / denominator : 1.0e-12 : 1.0e12;

const double y_1 = slope*( low_x - first_point[0] ) + first_point[1];
if(y_1 >= low_y - tolerance && y_1 <= high_y+tolerance) 
return true;
const double y_2 = slope*( high_x - first_point[0] ) + first_point[1];
if(y_2 >= low_y - tolerance && y_2 <= high_y+tolerance) 
return true;
const double x_1 = first_point[0] + ( (low_y - first_point[1]) / slope );
if(x_1 >= low_x-tolerance && x_1 <= high_x+tolerance) 
return true;
const double x_2 = first_point[0] + ( (high_y - first_point[1]) / slope );
if(x_2 >= low_x-tolerance && x_2 <= high_x+tolerance) 
return true;


return false;
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
+ std::pow(rPoint[1] - r_first_point[1], 2));

const double length_2 = std::sqrt( std::pow(rPoint[0] - r_second_point[0], 2)
+ std::pow(rPoint[1] - r_second_point[1], 2));

if (length_1 <= (length + tolerance) && length_2 <= (length + tolerance)) {
rResult[0] = 2.0 * length_1/(length + tolerance) - 1.0;
} else {
if (length_1 > length_2) {
rResult[0] = 2.0 * length_1/(length + tolerance) - 1.0;
} else {
rResult[0] = -2.0 * length_1/(length + tolerance) - 1.0;
}
}

return rResult ;
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

ProjectionPointGlobalToLocalSpace(rPointGlobalCoordinates, rProjectedPointLocalCoordinates, Tolerance);

this->GlobalCoordinates(rProjectedPointGlobalCoordinates, rProjectedPointLocalCoordinates);

return 1;
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
CoordinatesArrayType proj_pt_gl_coords;
GeometricalProjectionUtilities::FastProjectOnLine2D(*this, rPointGlobalCoordinates, proj_pt_gl_coords);

PointLocalCoordinates( rProjectionPointLocalCoordinates, proj_pt_gl_coords );

return 1;
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

Line2D2(): BaseType( PointsArrayType(), &msGeometryData ) {}




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
Matrix aux_mat = ZeroMatrix(2, 1);
aux_mat(0, 0) = -0.5;
aux_mat(1, 0) =  0.5;
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
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsValues( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients = {{
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_GAUSS_5 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_1 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_2 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_3 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_4 ),
Line2D2<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients( GeometryData::IntegrationMethod::GI_EXTENDED_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}






template<class TOtherPointType> friend class Line2D2;





}; 






template<class TPointType>
inline std::istream& operator >> ( std::istream& rIStream,
Line2D2<TPointType>& rThis );

template<class TPointType>
inline std::ostream& operator << ( std::ostream& rOStream,
const Line2D2<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}



template<class TPointType>
const GeometryData Line2D2<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
Line2D2<TPointType>::AllIntegrationPoints(),
Line2D2<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients() );

template<class TPointType>
const GeometryDimension Line2D2<TPointType>::msGeometryDimension(2, 1);

}  