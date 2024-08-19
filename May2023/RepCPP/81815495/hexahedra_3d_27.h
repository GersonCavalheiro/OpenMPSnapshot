
#pragma once



#include "geometries/quadrilateral_3d_9.h"
#include "utilities/integration_utilities.h"
#include "integration/hexahedron_gauss_legendre_integration_points.h"

namespace Kratos
{

template<class TPointType> class Hexahedra3D27 : public Geometry<TPointType>
{
public:



typedef Geometry<TPointType> BaseType;


typedef Line3D3<TPointType> EdgeType;
typedef Quadrilateral3D9<TPointType> FaceType;


KRATOS_CLASS_POINTER_DEFINITION( Hexahedra3D27 );


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




Hexahedra3D27( const PointType& Point1, const PointType& Point2, const PointType& Point3,
const PointType& Point4, const PointType& Point5, const PointType& Point6,
const PointType& Point7, const PointType& Point8, const PointType& Point9,
const PointType& Point10, const PointType& Point11, const PointType& Point12,
const PointType& Point13, const PointType& Point14, const PointType& Point15,
const PointType& Point16, const PointType& Point17, const PointType& Point18,
const PointType& Point19, const PointType& Point20, const PointType& Point21,
const PointType& Point22, const PointType& Point23, const PointType& Point24,
const PointType& Point25, const PointType& Point26, const PointType& Point27
)
: BaseType( PointsArrayType(), &msGeometryData )
{
this->Points().push_back( typename PointType::Pointer( new PointType( Point1 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point2 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point3 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point4 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point5 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point6 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point7 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point8 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point9 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point10 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point11 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point12 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point13 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point14 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point15 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point16 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point17 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point18 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point19 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point20 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point21 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point22 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point23 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point24 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point25 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point26 ) ) );
this->Points().push_back( typename PointType::Pointer( new PointType( Point27 ) ) );
}

Hexahedra3D27( typename PointType::Pointer pPoint1,
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
typename PointType::Pointer pPoint13,
typename PointType::Pointer pPoint14,
typename PointType::Pointer pPoint15,
typename PointType::Pointer pPoint16,
typename PointType::Pointer pPoint17,
typename PointType::Pointer pPoint18,
typename PointType::Pointer pPoint19,
typename PointType::Pointer pPoint20,
typename PointType::Pointer pPoint21,
typename PointType::Pointer pPoint22,
typename PointType::Pointer pPoint23,
typename PointType::Pointer pPoint24,
typename PointType::Pointer pPoint25,
typename PointType::Pointer pPoint26,
typename PointType::Pointer pPoint27 )
: BaseType( PointsArrayType(), &msGeometryData )
{
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
this->Points().push_back( pPoint11 );
this->Points().push_back( pPoint12 );
this->Points().push_back( pPoint13 );
this->Points().push_back( pPoint14 );
this->Points().push_back( pPoint15 );
this->Points().push_back( pPoint16 );
this->Points().push_back( pPoint17 );
this->Points().push_back( pPoint18 );
this->Points().push_back( pPoint19 );
this->Points().push_back( pPoint20 );
this->Points().push_back( pPoint21 );
this->Points().push_back( pPoint22 );
this->Points().push_back( pPoint23 );
this->Points().push_back( pPoint24 );
this->Points().push_back( pPoint25 );
this->Points().push_back( pPoint26 );
this->Points().push_back( pPoint27 );
}

Hexahedra3D27( const PointsArrayType& ThisPoints )
: BaseType( ThisPoints, &msGeometryData )
{
if ( this->PointsNumber() != 27 )
KRATOS_ERROR << "Invalid points number. Expected 27, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D27(
const IndexType GeometryId,
const PointsArrayType& rThisPoints
) : BaseType( GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF( this->PointsNumber() != 27 ) << "Invalid points number. Expected 27, given " << this->PointsNumber() << std::endl;
}

explicit Hexahedra3D27(
const std::string& rGeometryName,
const PointsArrayType& rThisPoints
) : BaseType( rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 27) << "Invalid points number. Expected 27, given " << this->PointsNumber() << std::endl;
}


Hexahedra3D27( Hexahedra3D27 const& rOther )
: BaseType( rOther )
{
}


template<class TOtherPointType> Hexahedra3D27( Hexahedra3D27<TOtherPointType> const& rOther )
: BaseType( rOther )
{
}

~Hexahedra3D27() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Hexahedra;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Hexahedra3D27;
}




Hexahedra3D27& operator=( const Hexahedra3D27& rOther )
{
BaseType::operator=( rOther );
return *this;
}


template<class TOtherPointType>
Hexahedra3D27& operator=( Hexahedra3D27<TOtherPointType> const & rOther )
{
BaseType::operator=( rOther );

return *this;
}





typename BaseType::Pointer Create(
IndexType NewId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new Hexahedra3D27( NewId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new Hexahedra3D27( NewGeometryId, rGeometry.Points() ) );
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
return Volume();
}


bool IsInside(
const CoordinatesArrayType& rPoint,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
this->PointLocalCoordinates( rResult, rPoint );

if ( std::abs( rResult[0] ) <= (1.0 + Tolerance) )
{
if ( std::abs( rResult[1] ) <= (1.0 + Tolerance) )
{
if ( std::abs( rResult[2] ) <= (1.0 + Tolerance) )
{
return true;
}
}
}

return false;
}



SizeType EdgesNumber() const override
{
return 12;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer EdgePointerType;
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 8 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 2 ),
this->pGetPoint( 9 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 10 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 11 ) ) ) );

edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ),
this->pGetPoint( 16 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 5 ),
this->pGetPoint( 6 ),
this->pGetPoint( 17 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 6 ),
this->pGetPoint( 7 ),
this->pGetPoint( 18 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 7 ),
this->pGetPoint( 4 ),
this->pGetPoint( 19 ) ) ) );

edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 0 ),
this->pGetPoint( 4 ),
this->pGetPoint( 12 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 1 ),
this->pGetPoint( 5 ),
this->pGetPoint( 13 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ),
this->pGetPoint( 14 ) ) ) );
edges.push_back( EdgePointerType( new EdgeType(
this->pGetPoint( 3 ),
this->pGetPoint( 7 ),
this->pGetPoint( 15 ) ) ) );
return edges;
}



SizeType FacesNumber() const override
{
return 6;
}


GeometriesArrayType GenerateFaces() const override
{
GeometriesArrayType faces = GeometriesArrayType();
typedef typename Geometry<TPointType>::Pointer FacePointerType;

faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 3 ),
this->pGetPoint( 2 ),
this->pGetPoint( 1 ),
this->pGetPoint( 0 ),
this->pGetPoint( 10 ),
this->pGetPoint( 9 ),
this->pGetPoint( 8 ),
this->pGetPoint( 11 ),
this->pGetPoint( 20 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 0 ),
this->pGetPoint( 1 ),
this->pGetPoint( 5 ),
this->pGetPoint( 4 ),
this->pGetPoint( 8 ),
this->pGetPoint( 13 ),
this->pGetPoint( 16 ),
this->pGetPoint( 12 ),
this->pGetPoint( 21 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 2 ),
this->pGetPoint( 6 ),
this->pGetPoint( 5 ),
this->pGetPoint( 1 ),
this->pGetPoint( 14 ),
this->pGetPoint( 17 ),
this->pGetPoint( 13 ),
this->pGetPoint( 9 ),
this->pGetPoint( 22 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 6 ),
this->pGetPoint( 2 ),
this->pGetPoint( 3 ),
this->pGetPoint( 14 ),
this->pGetPoint( 18 ),
this->pGetPoint( 10 ),
this->pGetPoint( 15 ),
this->pGetPoint( 23 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 7 ),
this->pGetPoint( 3 ),
this->pGetPoint( 0 ),
this->pGetPoint( 4 ),
this->pGetPoint( 15 ),
this->pGetPoint( 11 ),
this->pGetPoint( 12 ),
this->pGetPoint( 19 ),
this->pGetPoint( 24 ) ) ) );
faces.push_back( FacePointerType( new FaceType(
this->pGetPoint( 4 ),
this->pGetPoint( 5 ),
this->pGetPoint( 6 ),
this->pGetPoint( 7 ),
this->pGetPoint( 16 ),
this->pGetPoint( 17 ),
this->pGetPoint( 18 ),
this->pGetPoint( 19 ),
this->pGetPoint( 25 ) ) ) );
return faces;
}





double ShapeFunctionValue( IndexType ShapeFunctionIndex,
const CoordinatesArrayType& rPoint ) const override
{
double fx1 = 0.5 * ( rPoint[0] - 1.0 ) * ( rPoint[0] );
double fx2 = 0.5 * ( rPoint[0] + 1.0 ) * ( rPoint[0] );
double fx3 = 1.0 - ( rPoint[0] * rPoint[0] );
double fy1 = 0.5 * ( rPoint[1] - 1.0 ) * ( rPoint[1] );
double fy2 = 0.5 * ( rPoint[1] + 1.0 ) * ( rPoint[1] );
double fy3 = 1.0 - ( rPoint[1] * rPoint[1] );
double fz1 = 0.5 * ( rPoint[2] - 1.0 ) * ( rPoint[2] );
double fz2 = 0.5 * ( rPoint[2] + 1.0 ) * ( rPoint[2] );
double fz3 = 1.0 - ( rPoint[2] * rPoint[2] );

switch ( ShapeFunctionIndex )
{
case 0:
return( fx1*fy1*fz1 );
case 1:
return( fx2*fy1*fz1 );
case 2:
return( fx2*fy2*fz1 );
case 3:
return( fx1*fy2*fz1 );
case 4:
return( fx1*fy1*fz2 );
case 5:
return( fx2*fy1*fz2 );
case 6:
return( fx2*fy2*fz2 );
case 7:
return( fx1*fy2*fz2 );
case 8:
return( fx3*fy1*fz1 );
case 9:
return( fx2*fy3*fz1 );
case 10:
return( fx3*fy2*fz1 );
case 11:
return( fx1*fy3*fz1 );
case 12:
return( fx1*fy1*fz3 );
case 13:
return( fx2*fy1*fz3 );
case 14:
return( fx2*fy2*fz3 );
case 15:
return( fx1*fy2*fz3 );
case 16:
return( fx3*fy1*fz2 );
case 17:
return( fx2*fy3*fz2 );
case 18:
return( fx3*fy2*fz2 );
case 19:
return( fx1*fy3*fz2 );
case 20:
return( fx3*fy3*fz1 );
case 21:
return( fx3*fy1*fz3 );
case 22:
return( fx2*fy3*fz3 );
case 23:
return( fx3*fy2*fz3 );
case 24:
return( fx1*fy3*fz3 );
case 25:
return( fx3*fy3*fz2 );
case 26:
return( fx3*fy3*fz3 );

default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
}

return 0;
}


Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const override
{
if(rResult.size() != 27) rResult.resize(27,false);

double fx1 = 0.5 * ( rCoordinates[0] - 1.0 ) * ( rCoordinates[0] );
double fx2 = 0.5 * ( rCoordinates[0] + 1.0 ) * ( rCoordinates[0] );
double fx3 = 1.0 - ( rCoordinates[0] * rCoordinates[0] );
double fy1 = 0.5 * ( rCoordinates[1] - 1.0 ) * ( rCoordinates[1] );
double fy2 = 0.5 * ( rCoordinates[1] + 1.0 ) * ( rCoordinates[1] );
double fy3 = 1.0 - ( rCoordinates[1] * rCoordinates[1] );
double fz1 = 0.5 * ( rCoordinates[2] - 1.0 ) * ( rCoordinates[2] );
double fz2 = 0.5 * ( rCoordinates[2] + 1.0 ) * ( rCoordinates[2] );
double fz3 = 1.0 - ( rCoordinates[2] * rCoordinates[2] );

rResult[0] = ( fx1*fy1*fz1 );
rResult[1] = ( fx2*fy1*fz1 );
rResult[2] = ( fx2*fy2*fz1 );
rResult[3] = ( fx1*fy2*fz1 );
rResult[4] = ( fx1*fy1*fz2 );
rResult[5] = ( fx2*fy1*fz2 );
rResult[6] = ( fx2*fy2*fz2 );
rResult[7] = ( fx1*fy2*fz2 );
rResult[8] = ( fx3*fy1*fz1 );
rResult[9] = ( fx2*fy3*fz1 );
rResult[10] = ( fx3*fy2*fz1 );
rResult[11] = ( fx1*fy3*fz1 );
rResult[12] = ( fx1*fy1*fz3 );
rResult[13] = ( fx2*fy1*fz3 );
rResult[14] = ( fx2*fy2*fz3 );
rResult[15] = ( fx1*fy2*fz3 );
rResult[16] = ( fx3*fy1*fz2 );
rResult[17] = ( fx2*fy3*fz2 );
rResult[18] = ( fx3*fy2*fz2 );
rResult[19] = ( fx1*fy3*fz2 );
rResult[20] = ( fx3*fy3*fz1 );
rResult[21] = ( fx3*fy1*fz3 );
rResult[22] = ( fx2*fy3*fz3 );
rResult[23] = ( fx3*fy2*fz3 );
rResult[24] = ( fx1*fy3*fz3 );
rResult[25] = ( fx3*fy3*fz2 );
rResult[26] = ( fx3*fy3*fz3 );

return rResult;
}



std::string Info() const override
{
return "3 dimensional hexahedra with 27 nodes and quadratic shape functions in 3D space";
}


void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "3 dimensional hexahedra with 27 nodes and quadratic shape functions in 3D space";
}


void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
Matrix jacobian;
this->Jacobian( jacobian, PointType() );
rOStream << "Jacobian in the origin\t : " << jacobian;
}



Matrix& ShapeFunctionsLocalGradients( Matrix& result,
const CoordinatesArrayType& rPoint ) const override
{
double fx1 = 0.5 * ( rPoint[0] - 1.0 ) * ( rPoint[0] );
double fx2 = 0.5 * ( rPoint[0] + 1.0 ) * ( rPoint[0] );
double fx3 = 1.0 - ( rPoint[0] * rPoint[0] );
double fy1 = 0.5 * ( rPoint[1] - 1.0 ) * ( rPoint[1] );
double fy2 = 0.5 * ( rPoint[1] + 1.0 ) * ( rPoint[1] );
double fy3 = 1.0 - ( rPoint[1] * rPoint[1] );
double fz1 = 0.5 * ( rPoint[2] - 1.0 ) * ( rPoint[2] );
double fz2 = 0.5 * ( rPoint[2] + 1.0 ) * ( rPoint[2] );
double fz3 = 1.0 - ( rPoint[2] * rPoint[2] );

double gx1 = 0.5 * ( 2.0 * rPoint[0] - 1.0 );
double gx2 = 0.5 * ( 2.0 * rPoint[0] + 1.0 );
double gx3 = -2 * rPoint[0];
double gy1 = 0.5 * ( 2.0 * rPoint[1] - 1.0 );
double gy2 = 0.5 * ( 2.0 * rPoint[1] + 1.0 );
double gy3 = -2 * rPoint[1];
double gz1 = 0.5 * ( 2.0 * rPoint[2] - 1.0 );
double gz2 = 0.5 * ( 2.0 * rPoint[2] + 1.0 );
double gz3 = -2 * rPoint[2];

if ( result.size1() != 27 || result.size2() != 3 )
result.resize( 27, 3, false );

result( 0, 0 ) = gx1 * fy1 * fz1;

result( 0, 1 ) = fx1 * gy1 * fz1;

result( 0, 2 ) = fx1 * fy1 * gz1;

result( 1, 0 ) = gx2 * fy1 * fz1;

result( 1, 1 ) = fx2 * gy1 * fz1;

result( 1, 2 ) = fx2 * fy1 * gz1;

result( 2, 0 ) = gx2 * fy2 * fz1;

result( 2, 1 ) = fx2 * gy2 * fz1;

result( 2, 2 ) = fx2 * fy2 * gz1;

result( 3, 0 ) = gx1 * fy2 * fz1;

result( 3, 1 ) = fx1 * gy2 * fz1;

result( 3, 2 ) = fx1 * fy2 * gz1;

result( 4, 0 ) = gx1 * fy1 * fz2;

result( 4, 1 ) = fx1 * gy1 * fz2;

result( 4, 2 ) = fx1 * fy1 * gz2;

result( 5, 0 ) = gx2 * fy1 * fz2;

result( 5, 1 ) = fx2 * gy1 * fz2;

result( 5, 2 ) = fx2 * fy1 * gz2;

result( 6, 0 ) = gx2 * fy2 * fz2;

result( 6, 1 ) = fx2 * gy2 * fz2;

result( 6, 2 ) = fx2 * fy2 * gz2;

result( 7, 0 ) = gx1 * fy2 * fz2;

result( 7, 1 ) = fx1 * gy2 * fz2;

result( 7, 2 ) = fx1 * fy2 * gz2;

result( 8, 0 ) = gx3 * fy1 * fz1;

result( 8, 1 ) = fx3 * gy1 * fz1;

result( 8, 2 ) = fx3 * fy1 * gz1;

result( 9, 0 ) = gx2 * fy3 * fz1;

result( 9, 1 ) = fx2 * gy3 * fz1;

result( 9, 2 ) = fx2 * fy3 * gz1;

result( 10, 0 ) = gx3 * fy2 * fz1;

result( 10, 1 ) = fx3 * gy2 * fz1;

result( 10, 2 ) = fx3 * fy2 * gz1;

result( 11, 0 ) = gx1 * fy3 * fz1;

result( 11, 1 ) = fx1 * gy3 * fz1;

result( 11, 2 ) = fx1 * fy3 * gz1;

result( 12, 0 ) = gx1 * fy1 * fz3;

result( 12, 1 ) = fx1 * gy1 * fz3;

result( 12, 2 ) = fx1 * fy1 * gz3;

result( 13, 0 ) = gx2 * fy1 * fz3;

result( 13, 1 ) = fx2 * gy1 * fz3;

result( 13, 2 ) = fx2 * fy1 * gz3;

result( 14, 0 ) = gx2 * fy2 * fz3;

result( 14, 1 ) = fx2 * gy2 * fz3;

result( 14, 2 ) = fx2 * fy2 * gz3;

result( 15, 0 ) = gx1 * fy2 * fz3;

result( 15, 1 ) = fx1 * gy2 * fz3;

result( 15, 2 ) = fx1 * fy2 * gz3;

result( 16, 0 ) = gx3 * fy1 * fz2;

result( 16, 1 ) = fx3 * gy1 * fz2;

result( 16, 2 ) = fx3 * fy1 * gz2;

result( 17, 0 ) = gx2 * fy3 * fz2;

result( 17, 1 ) = fx2 * gy3 * fz2;

result( 17, 2 ) = fx2 * fy3 * gz2;

result( 18, 0 ) = gx3 * fy2 * fz2;

result( 18, 1 ) = fx3 * gy2 * fz2;

result( 18, 2 ) = fx3 * fy2 * gz2;

result( 19, 0 ) = gx1 * fy3 * fz2;

result( 19, 1 ) = fx1 * gy3 * fz2;

result( 19, 2 ) = fx1 * fy3 * gz2;

result( 20, 0 ) = gx3 * fy3 * fz1;

result( 20, 1 ) = fx3 * gy3 * fz1;

result( 20, 2 ) = fx3 * fy3 * gz1;

result( 21, 0 ) = gx3 * fy1 * fz3;

result( 21, 1 ) = fx3 * gy1 * fz3;

result( 21, 2 ) = fx3 * fy1 * gz3;

result( 22, 0 ) = gx2 * fy3 * fz3;

result( 22, 1 ) = fx2 * gy3 * fz3;

result( 22, 2 ) = fx2 * fy3 * gz3;

result( 23, 0 ) = gx3 * fy2 * fz3;

result( 23, 1 ) = fx3 * gy2 * fz3;

result( 23, 2 ) = fx3 * fy2 * gz3;

result( 24, 0 ) = gx1 * fy3 * fz3;

result( 24, 1 ) = fx1 * gy3 * fz3;

result( 24, 2 ) = fx1 * fy3 * gz3;

result( 25, 0 ) = gx3 * fy3 * fz2;

result( 25, 1 ) = fx3 * gy3 * fz2;

result( 25, 2 ) = fx3 * fy3 * gz2;

result( 26, 0 ) = gx3 * fy3 * fz3;

result( 26, 1 ) = fx3 * gy3 * fz3;

result( 26, 2 ) = fx3 * fy3 * gz3;

return( result );
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

Hexahedra3D27(): BaseType( PointsArrayType(), &msGeometryData ) {}







static Matrix CalculateShapeFunctionsIntegrationPointsValues(
typename BaseType::IntegrationMethod ThisMethod )
{
IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 27;
Matrix shape_function_values( integration_points_number, points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
double fx1 = 0.5 * ( integration_points[pnt].X() - 1.0 ) * ( integration_points[pnt].X() );
double fx2 = 0.5 * ( integration_points[pnt].X() + 1.0 ) * ( integration_points[pnt].X() );
double fx3 = 1.0 - ( integration_points[pnt].X() * integration_points[pnt].X() );
double fy1 = 0.5 * ( integration_points[pnt].Y() - 1.0 ) * ( integration_points[pnt].Y() );
double fy2 = 0.5 * ( integration_points[pnt].Y() + 1.0 ) * ( integration_points[pnt].Y() );
double fy3 = 1.0 - ( integration_points[pnt].Y() * integration_points[pnt].Y() );
double fz1 = 0.5 * ( integration_points[pnt].Z() - 1.0 ) * ( integration_points[pnt].Z() );
double fz2 = 0.5 * ( integration_points[pnt].Z() + 1.0 ) * ( integration_points[pnt].Z() );
double fz3 = 1.0 - ( integration_points[pnt].Z() * integration_points[pnt].Z() );

shape_function_values( pnt, 0 ) = ( fx1 * fy1 * fz1 );
shape_function_values( pnt, 1 ) = ( fx2 * fy1 * fz1 );
shape_function_values( pnt, 2 ) = ( fx2 * fy2 * fz1 );
shape_function_values( pnt, 3 ) = ( fx1 * fy2 * fz1 );
shape_function_values( pnt, 4 ) = ( fx1 * fy1 * fz2 );
shape_function_values( pnt, 5 ) = ( fx2 * fy1 * fz2 );
shape_function_values( pnt, 6 ) = ( fx2 * fy2 * fz2 );
shape_function_values( pnt, 7 ) = ( fx1 * fy2 * fz2 );
shape_function_values( pnt, 8 ) = ( fx3 * fy1 * fz1 );
shape_function_values( pnt, 9 ) = ( fx2 * fy3 * fz1 );
shape_function_values( pnt, 10 ) = ( fx3 * fy2 * fz1 );
shape_function_values( pnt, 11 ) = ( fx1 * fy3 * fz1 );
shape_function_values( pnt, 12 ) = ( fx1 * fy1 * fz3 );
shape_function_values( pnt, 13 ) = ( fx2 * fy1 * fz3 );
shape_function_values( pnt, 14 ) = ( fx2 * fy2 * fz3 );
shape_function_values( pnt, 15 ) = ( fx1 * fy2 * fz3 );
shape_function_values( pnt, 16 ) = ( fx3 * fy1 * fz2 );
shape_function_values( pnt, 17 ) = ( fx2 * fy3 * fz2 );
shape_function_values( pnt, 18 ) = ( fx3 * fy2 * fz2 );
shape_function_values( pnt, 19 ) = ( fx1 * fy3 * fz2 );
shape_function_values( pnt, 20 ) = ( fx3 * fy3 * fz1 );
shape_function_values( pnt, 21 ) = ( fx3 * fy1 * fz3 );
shape_function_values( pnt, 22 ) = ( fx2 * fy3 * fz3 );
shape_function_values( pnt, 23 ) = ( fx3 * fy2 * fz3 );
shape_function_values( pnt, 24 ) = ( fx1 * fy3 * fz3 );
shape_function_values( pnt, 25 ) = ( fx3 * fy3 * fz2 );
shape_function_values( pnt, 26 ) = ( fx3 * fy3 * fz3 );
}

return shape_function_values;
}



static ShapeFunctionsGradientsType
CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod )
{
const IntegrationPointsContainerType all_integration_points =
AllIntegrationPoints();
const IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values( integration_points_number );

for ( int pnt = 0; pnt < integration_points_number; pnt++ )
{
double fx1 = 0.5 * ( integration_points[pnt].X() - 1.0 ) * ( integration_points[pnt].X() );
double fx2 = 0.5 * ( integration_points[pnt].X() + 1.0 ) * ( integration_points[pnt].X() );
double fx3 = 1.0 - ( integration_points[pnt].X() * integration_points[pnt].X() );
double fy1 = 0.5 * ( integration_points[pnt].Y() - 1.0 ) * ( integration_points[pnt].Y() );
double fy2 = 0.5 * ( integration_points[pnt].Y() + 1.0 ) * ( integration_points[pnt].Y() );
double fy3 = 1.0 - ( integration_points[pnt].Y() * integration_points[pnt].Y() );
double fz1 = 0.5 * ( integration_points[pnt].Z() - 1.0 ) * ( integration_points[pnt].Z() );
double fz2 = 0.5 * ( integration_points[pnt].Z() + 1.0 ) * ( integration_points[pnt].Z() );
double fz3 = 1.0 - ( integration_points[pnt].Z() * integration_points[pnt].Z() );

double gx1 = 0.5 * ( 2.0 * integration_points[pnt].X() - 1.0 );
double gx2 = 0.5 * ( 2.0 * integration_points[pnt].X() + 1.0 );
double gx3 = -2 * integration_points[pnt].X();
double gy1 = 0.5 * ( 2.0 * integration_points[pnt].Y() - 1.0 );
double gy2 = 0.5 * ( 2.0 * integration_points[pnt].Y() + 1.0 );
double gy3 = -2 * integration_points[pnt].Y();
double gz1 = 0.5 * ( 2.0 * integration_points[pnt].Z() - 1.0 );
double gz2 = 0.5 * ( 2.0 * integration_points[pnt].Z() + 1.0 );
double gz3 = -2 * integration_points[pnt].Z();
Matrix result = ZeroMatrix( 27, 3 );

result( 0, 0 ) = gx1 * fy1 * fz1;
result( 0, 1 ) = fx1 * gy1 * fz1;
result( 0, 2 ) = fx1 * fy1 * gz1;

result( 1, 0 ) = gx2 * fy1 * fz1;
result( 1, 1 ) = fx2 * gy1 * fz1;
result( 1, 2 ) = fx2 * fy1 * gz1;

result( 2, 0 ) = gx2 * fy2 * fz1;
result( 2, 1 ) = fx2 * gy2 * fz1;
result( 2, 2 ) = fx2 * fy2 * gz1;

result( 3, 0 ) = gx1 * fy2 * fz1;
result( 3, 1 ) = fx1 * gy2 * fz1;
result( 3, 2 ) = fx1 * fy2 * gz1;

result( 4, 0 ) = gx1 * fy1 * fz2;
result( 4, 1 ) = fx1 * gy1 * fz2;
result( 4, 2 ) = fx1 * fy1 * gz2;

result( 5, 0 ) = gx2 * fy1 * fz2;
result( 5, 1 ) = fx2 * gy1 * fz2;
result( 5, 2 ) = fx2 * fy1 * gz2;

result( 6, 0 ) = gx2 * fy2 * fz2;
result( 6, 1 ) = fx2 * gy2 * fz2;
result( 6, 2 ) = fx2 * fy2 * gz2;

result( 7, 0 ) = gx1 * fy2 * fz2;
result( 7, 1 ) = fx1 * gy2 * fz2;
result( 7, 2 ) = fx1 * fy2 * gz2;

result( 8, 0 ) = gx3 * fy1 * fz1;
result( 8, 1 ) = fx3 * gy1 * fz1;
result( 8, 2 ) = fx3 * fy1 * gz1;

result( 9, 0 ) = gx2 * fy3 * fz1;
result( 9, 1 ) = fx2 * gy3 * fz1;
result( 9, 2 ) = fx2 * fy3 * gz1;

result( 10, 0 ) = gx3 * fy2 * fz1;
result( 10, 1 ) = fx3 * gy2 * fz1;
result( 10, 2 ) = fx3 * fy2 * gz1;

result( 11, 0 ) = gx1 * fy3 * fz1;
result( 11, 1 ) = fx1 * gy3 * fz1;
result( 11, 2 ) = fx1 * fy3 * gz1;

result( 12, 0 ) = gx1 * fy1 * fz3;
result( 12, 1 ) = fx1 * gy1 * fz3;
result( 12, 2 ) = fx1 * fy1 * gz3;

result( 13, 0 ) = gx2 * fy1 * fz3;
result( 13, 1 ) = fx2 * gy1 * fz3;
result( 13, 2 ) = fx2 * fy1 * gz3;

result( 14, 0 ) = gx2 * fy2 * fz3;
result( 14, 1 ) = fx2 * gy2 * fz3;
result( 14, 2 ) = fx2 * fy2 * gz3;

result( 15, 0 ) = gx1 * fy2 * fz3;
result( 15, 1 ) = fx1 * gy2 * fz3;
result( 15, 2 ) = fx1 * fy2 * gz3;

result( 16, 0 ) = gx3 * fy1 * fz2;
result( 16, 1 ) = fx3 * gy1 * fz2;
result( 16, 2 ) = fx3 * fy1 * gz2;

result( 17, 0 ) = gx2 * fy3 * fz2;
result( 17, 1 ) = fx2 * gy3 * fz2;
result( 17, 2 ) = fx2 * fy3 * gz2;

result( 18, 0 ) = gx3 * fy2 * fz2;
result( 18, 1 ) = fx3 * gy2 * fz2;
result( 18, 2 ) = fx3 * fy2 * gz2;

result( 19, 0 ) = gx1 * fy3 * fz2;
result( 19, 1 ) = fx1 * gy3 * fz2;
result( 19, 2 ) = fx1 * fy3 * gz2;

result( 20, 0 ) = gx3 * fy3 * fz1;
result( 20, 1 ) = fx3 * gy3 * fz1;
result( 20, 2 ) = fx3 * fy3 * gz1;

result( 21, 0 ) = gx3 * fy1 * fz3;
result( 21, 1 ) = fx3 * gy1 * fz3;
result( 21, 2 ) = fx3 * fy1 * gz3;

result( 22, 0 ) = gx2 * fy3 * fz3;
result( 22, 1 ) = fx2 * gy3 * fz3;
result( 22, 2 ) = fx2 * fy3 * gz3;

result( 23, 0 ) = gx3 * fy2 * fz3;
result( 23, 1 ) = fx3 * gy2 * fz3;
result( 23, 2 ) = fx3 * fy2 * gz3;

result( 24, 0 ) = gx1 * fy3 * fz3;
result( 24, 1 ) = fx1 * gy3 * fz3;
result( 24, 2 ) = fx1 * fy3 * gz3;

result( 25, 0 ) = gx3 * fy3 * fz2;
result( 25, 1 ) = fx3 * gy3 * fz2;
result( 25, 2 ) = fx3 * fy3 * gz2;

result( 26, 0 ) = gx3 * fy3 * fz3;
result( 26, 1 ) = fx3 * gy3 * fz3;
result( 26, 2 ) = fx3 * fy3 * gz3;

d_shape_f_values[pnt] = result;
}

return d_shape_f_values;
}


static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature < HexahedronGaussLegendreIntegrationPoints1,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints2,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints3,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints4,
3, IntegrationPoint<3> >::GenerateIntegrationPoints(),
Quadrature < HexahedronGaussLegendreIntegrationPoints5,
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
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(
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
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_1 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_2 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_3 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_4 ),
Hexahedra3D27<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(
GeometryData::IntegrationMethod::GI_GAUSS_5 )
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Hexahedra3D27;




};





template<class TPointType> inline std::istream& operator >> (
std::istream& rIStream, Hexahedra3D27<TPointType>& rThis );


template<class TPointType> inline std::ostream& operator << (
std::ostream& rOStream, const Hexahedra3D27<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}

template<class TPointType> const
GeometryData Hexahedra3D27<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_3,
Hexahedra3D27<TPointType>::AllIntegrationPoints(),
Hexahedra3D27<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients()
);

template<class TPointType>
const GeometryDimension Hexahedra3D27<TPointType>::msGeometryDimension(3, 3);

}