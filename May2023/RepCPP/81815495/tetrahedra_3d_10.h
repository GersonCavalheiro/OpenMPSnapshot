
#pragma once

#include <numeric>


#include "geometries/triangle_3d_6.h"
#include "geometries/tetrahedra_3d_4.h"
#include "utilities/integration_utilities.h"
#include "integration/tetrahedron_gauss_legendre_integration_points.h"

namespace Kratos
{

template<class TPointType> class Tetrahedra3D10 : public Geometry<TPointType>
{
public:



typedef Geometry<TPointType> BaseType;


typedef Line3D3<TPointType> EdgeType;
typedef Triangle3D6<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Tetrahedra3D10 );


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





Tetrahedra3D10( typename PointType::Pointer pPoint1,
typename PointType::Pointer pPoint2,
typename PointType::Pointer pPoint3,
typename PointType::Pointer pPoint4,
typename PointType::Pointer pPoint5,
typename PointType::Pointer pPoint6,
typename PointType::Pointer pPoint7,
typename PointType::Pointer pPoint8,
typename PointType::Pointer pPoint9,
typename PointType::Pointer pPoint10
)
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().reserve( 10 );
this->Points().push_back( pPoint1 );
this->Points().push_back( pPoint2 );
this->Points().push_back( pPoint3 );
this->Points().push_back( pPoint4 );
this->Points().push_back( pPoint5 );
this->Points().push_back( pPoint6 );
this->Points().push_back( pPoint7 );
this->Points().push_back( pPoint8 );
this->Points().push_back( pPoint9 );
this->Points().push_back( pPoint10 );
}

Tetrahedra3D10( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 10 )
KRATOS_ERROR << "Invalid points number. Expected 10, given " << this->PointsNumber() << std::endl;
}

explicit Tetrahedra3D10(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 10 ) << "Invalid points number. Expected 10, given " << this->PointsNumber() << std::endl;
}

explicit Tetrahedra3D10(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 10) << "Invalid points number. Expected 10, given " << this->PointsNumber() << std::endl;
}


Tetrahedra3D10( Tetrahedra3D10 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Tetrahedra3D10( Tetrahedra3D10<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Tetrahedra3D10() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Tetrahedra;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Tetrahedra3D10;
}




Tetrahedra3D10& operator=( const Tetrahedra3D10& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Tetrahedra3D10& operator=( Tetrahedra3D10<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}




typename BaseType::Pointer Create(
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Tetrahedra3D10( rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Tetrahedra3D10( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Tetrahedra3D10( rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Tetrahedra3D10( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}



double Length() const override
{
return sqrt( fabs( this->DeterminantOfJacobian( PointType() ) ) );
}


double Area() const override
{
return Volume();
}


double Volume() const override
{
return IntegrationUtilities::ComputeVolume3DGeometry(*this);
}


double DomainSize() const override
{
return  Volume();
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( (rResult[0] >= (0.0 - Tolerance)) && (rResult[0] <= (1.0 + Tolerance)) )
{
if ( (rResult[1] >= (0.0 - Tolerance)) && (rResult[1] <= (1.0 + Tolerance)) )
{
if ( (rResult[2] >= (0.0 - Tolerance)) && (rResult[2] <= (1.0 + Tolerance)) )
{
if ((( 1.0 - ( rResult[0] + rResult[1] + rResult[2] ) ) >= (0.0 - Tolerance) ) && (( 1.0 - ( rResult[0] + rResult[1] + rResult[2] ) ) <= (1.0 + Tolerance) ) )
{
return true;
}
}
}
}

return false;
}



SizeType EdgesNumber() const override
{
return 6;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer EdgePointerType;
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 4 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 5 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 0 ),
this->pGetPoint( 6 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 3 ),
this->pGetPoint( 7 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 3 ),
this->pGetPoint( 8 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 9 ) ) ) );

return edges;
}



SizeType FacesNumber() const override
{
return 4;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer FacePointerType;
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 2 ),
this->pGetPoint( 1 ),
this->pGetPoint( 6 ),
this->pGetPoint( 5 ),
this->pGetPoint( 4 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 3 ),
this->pGetPoint( 2 ),
this->pGetPoint( 7 ),
this->pGetPoint( 9 ),
this->pGetPoint( 6 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 3 ),
this->pGetPoint( 4 ),
this->pGetPoint( 8 ),
this->pGetPoint( 7 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 1 ),
this->pGetPoint( 9 ),
this->pGetPoint( 8 ),
this->pGetPoint( 5 ) ) ) );
return faces;
}


double AverageEdgeLength() const override {
const GeometriesArrayType edges = this->GenerateEdges();
return std::accumulate(
edges.begin(),
edges.end(),
0.0,
[](double sum, const auto& rEdge) {return sum + rEdge.Length();}
) * 0.16666666666666666667;
}

Matrix& PointsLocalCoordinates( Matrix& rResult ) const override
{
if(rResult.size1()!= 10 || rResult.size2()!= 3)
rResult.resize(10, 3, false);
rResult(0,0)=0.0;
rResult(0,1)=0.0;
rResult(0,2)=0.0;
rResult(1,0)=1.0;
rResult(1,1)=0.0;
rResult(1,2)=0.0;
rResult(2,0)=0.0;
rResult(2,1)=1.0;
rResult(2,2)=0.0;
rResult(3,0)=0.0;
rResult(3,1)=0.0;
rResult(3,2)=1.0;
rResult(4,0)=0.5;
rResult(4,1)=0.0;
rResult(4,2)=0.0;
rResult(5,0)=0.5;
rResult(5,1)=0.5;
rResult(5,2)=0.0;
rResult(6,0)=0.0;
rResult(6,1)=0.5;
rResult(6,2)=0.0;
rResult(7,0)=0.0;
rResult(7,1)=0.0;
rResult(7,2)=0.5;
rResult(8,0)=0.5;
rResult(8,1)=0.0;
rResult(8,2)=0.5;
rResult(9,0)=0.0;
rResult(9,1)=0.5;
rResult(9,2)=0.5;
return rResult;
}




Vector& ShapeFunctionsValues(Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
ShapeFunctionsValuesImpl(rResult, rCoordinates);
return rResult;
}


double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
double fourthCoord = 1.0 - ( rPoint[0] + rPoint[1] + rPoint[2] );

switch ( ShapeFunctionIndex )
{
case 0:
return( fourthCoord*( 2*fourthCoord - 1 ) );
case 1:
return( rPoint[0]*( 2*rPoint[0] - 1 ) );
case 2:
return( rPoint[1]*( 2*rPoint[1] - 1 ) );
case 3:
return( rPoint[2]*( 2*rPoint[2] - 1 ) );
case 4:
return( 4*fourthCoord*rPoint[0] );
case 5:
return( 4*rPoint[0]*rPoint[1] );
case 6:
return( 4*fourthCoord*rPoint[1] );
case 7:
return( 4*fourthCoord*rPoint[2] );
case 8:
return( 4*rPoint[0]*rPoint[2] );
case 9:
return( 4*rPoint[1]*rPoint[2] );
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}

Matrix& ShapeFunctionsLocalGradients(Matrix& rResult,
const CoordinatesArrayType& rPoint) const override
{
const double fourthCoord = 1.0 - (rPoint[0] + rPoint[1] + rPoint[2]);
if (rResult.size1() != this->size() || rResult.size2() != this->LocalSpaceDimension())
rResult.resize(this->size(), this->LocalSpaceDimension(), false);

rResult(0, 0) = -(4.0 * fourthCoord - 1.0);
rResult(0, 1) = -(4.0 * fourthCoord - 1.0);
rResult(0, 2) = -(4.0 * fourthCoord - 1.0);
rResult(1, 0) =  4.0 * rPoint[0] - 1.0;
rResult(1, 1) =  0.0;
rResult(1, 2) =  0.0;
rResult(2, 0) =  0.0;
rResult(2, 1) =  4.0 * rPoint[1] - 1.0;
rResult(2, 2) =  0.0;
rResult(3, 0) =  0.0;
rResult(3, 1) =  0.0;
rResult(3, 2) =  4.0 * rPoint[2] - 1.0;
rResult(4, 0) = -4.0 * rPoint[0] + 4.0 * fourthCoord;
rResult(4, 1) = -4.0 * rPoint[0];
rResult(4, 2) = -4.0 * rPoint[0];
rResult(5, 0) =  4.0 * rPoint[1];
rResult(5, 1) =  4.0 * rPoint[0];
rResult(5, 2) =  0.0;
rResult(6, 0) = -4.0 * rPoint[1];
rResult(6, 1) = -4.0 * rPoint[1] + 4.0 * fourthCoord;
rResult(6, 2) = -4.0 * rPoint[1];
rResult(7, 0) = -4.0 * rPoint[2];
rResult(7, 1) = -4.0 * rPoint[2];
rResult(7, 2) = -4.0 * rPoint[2] + 4.0 * fourthCoord;
rResult(8, 0) =  4.0 * rPoint[2];
rResult(8, 1) =  0.0;
rResult(8, 2) =  4.0 * rPoint[0];
rResult(9, 0) =  0.0;
rResult(9, 1) =  4.0 * rPoint[2];
rResult(9, 2) =  4.0 * rPoint[1];

return rResult;
}


bool HasIntersection(const Point& rLowPoint, const Point& rHighPoint) const override
{
if (this->FacesArePlanar()) {
return Tetrahedra3D4<TPointType>(
this->pGetPoint(0),
this->pGetPoint(1),
this->pGetPoint(2),
this->pGetPoint(3)).HasIntersection(rLowPoint, rHighPoint);
} else {
KRATOS_ERROR << "\"HasIntersection\" is not implemented for non-planar 10 noded tetrahedra.";
}
return false;
}





std::string Info() const override
{
return "3 dimensional tetrahedra with ten nodes in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "3 dimensional tetrahedra with ten nodes in 3D space";
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

Tetrahedra3D10(): BaseType( PointsArrayType(), &msGeometryData ) {}





static void ShapeFunctionsValuesImpl(Vector &rResult, const CoordinatesArrayType& rCoordinates)
{
if (rResult.size() != 10)
rResult.resize(10, false);
const double fourthCoord = 1.0 - rCoordinates[0] - rCoordinates[1] - rCoordinates[2];
rResult[0] = fourthCoord * (2.0 * fourthCoord - 1.0);
rResult[1] = rCoordinates[0] * (2.0 * rCoordinates[0] - 1.0);
rResult[2] = rCoordinates[1] * (2.0 * rCoordinates[1] - 1.0);
rResult[3] = rCoordinates[2] * (2.0 * rCoordinates[2] - 1.0);
rResult[4] = 4.0 * fourthCoord * rCoordinates[0];
rResult[5] = 4.0 * rCoordinates[0] * rCoordinates[1];
rResult[6] = 4.0 * rCoordinates[1] * fourthCoord;
rResult[7] = 4.0 * rCoordinates[2] * fourthCoord;
rResult[8] = 4.0 * rCoordinates[0] * rCoordinates[2];
rResult[9] = 4.0 * rCoordinates[1] * rCoordinates[2];
}


static Matrix CalculateShapeFunctionsIntegrationPointsValues(typename BaseType::IntegrationMethod ThisMethod)
{
const std::size_t points_number = 10;
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
Matrix shape_function_values(integration_points.size(), points_number);
Vector N(points_number);
for (std::size_t pnt = 0; pnt < integration_points.size(); ++pnt)
{
ShapeFunctionsValuesImpl(N, integration_points[pnt]);
for (std::size_t i = 0; i < N.size(); ++i)
shape_function_values(pnt, i) = N[i];
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
double fourthCoord = 1.0 - ( integration_points[pnt].X() + integration_points[pnt].Y() + integration_points[pnt].Z() );
double fourthCoord_DX = -1.0;
double fourthCoord_DY = -1.0;
double fourthCoord_DZ = -1.0;

Matrix result = ZeroMatrix( 10, 3 );
result( 0, 0 ) = ( 4 * fourthCoord - 1.0 ) * fourthCoord_DX;
result( 0, 1 ) = ( 4 * fourthCoord - 1.0 ) * fourthCoord_DY;
result( 0, 2 ) = ( 4 * fourthCoord - 1.0 ) * fourthCoord_DZ;
result( 1, 0 ) =  4 * integration_points[pnt].X() - 1.0;
result( 1, 1 ) =  0.0;
result( 1, 2 ) =  0.0;
result( 2, 0 ) =  0.0;
result( 2, 1 ) =  4 * integration_points[pnt].Y() - 1.0;
result( 2, 2 ) =  0.0;
result( 3, 0 ) =  0.0;
result( 3, 1 ) =  0.0;
result( 3, 2 ) =  4 * integration_points[pnt].Z() - 1.0 ;
result( 4, 0 ) =  4 * fourthCoord_DX * integration_points[pnt].X() + 4 * fourthCoord;
result( 4, 1 ) =  4 * fourthCoord_DY * integration_points[pnt].X();
result( 4, 2 ) =  4 * fourthCoord_DZ * integration_points[pnt].X();
result( 5, 0 ) =  4 * integration_points[pnt].Y();
result( 5, 1 ) =  4 * integration_points[pnt].X();
result( 5, 2 ) =  0.0;
result( 6, 0 ) =  4 * fourthCoord_DX * integration_points[pnt].Y();
result( 6, 1 ) =  4 * fourthCoord_DY * integration_points[pnt].Y() + 4 * fourthCoord;
result( 6, 2 ) =  4 * fourthCoord_DZ * integration_points[pnt].Y();
result( 7, 0 ) =  4 * fourthCoord_DX * integration_points[pnt].Z();
result( 7, 1 ) =  4 * fourthCoord_DY * integration_points[pnt].Z();
result( 7, 2 ) =  4 * fourthCoord_DZ * integration_points[pnt].Z() + 4 * fourthCoord;
result( 8, 0 ) =  4 * integration_points[pnt].Z();
result( 8, 1 ) =  0.0;
result( 8, 2 ) =  4 * integration_points[pnt].X();
result( 9, 0 ) =  0.0;
result( 9, 1 ) =  4 * integration_points[pnt].Z();
result( 9, 2 ) =  4 * integration_points[pnt].Y();

d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < TetrahedronGaussLegendreIntegrationPoints1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < TetrahedronGaussLegendreIntegrationPoints2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < TetrahedronGaussLegendreIntegrationPoints3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < TetrahedronGaussLegendreIntegrationPoints4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < TetrahedronGaussLegendreIntegrationPoints5,
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
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Tetrahedra3D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}


bool FacesArePlanar() const
{
constexpr double tol = 1e-6;
for (auto& r_edge : this->GenerateEdges()) {
const double a = MathUtils<double>::Norm3(r_edge.GetPoint(0)-r_edge.GetPoint(1));
const double b = MathUtils<double>::Norm3(r_edge.GetPoint(1)-r_edge.GetPoint(2));
const double c = MathUtils<double>::Norm3(r_edge.GetPoint(2)-r_edge.GetPoint(0));
if (b + c > a*(1.0+tol) ) {
return false;
}
}
return true;
}




template<class TOtherPointType> friend class Tetrahedra3D10;





};





template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream, Tetrahedra3D10<TPointType>& rThis );


template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream, const Tetrahedra3D10<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}


template<class TPointType> const
GeometryData Tetrahedra3D10<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_2,
Tetrahedra3D10<TPointType>::AllIntegrationPoints(),
Tetrahedra3D10<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType> const
GeometryDimension Tetrahedra3D10<TPointType>::msGeometryDimension(3, 3);

}
