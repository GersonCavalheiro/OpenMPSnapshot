
#pragma once



#include "geometries/line_2d_4.h"
#include "integration/triangle_gauss_legendre_integration_points.h"

namespace Kratos
{






template<class TPointType>
class Triangle2D10 : public Geometry<TPointType>
{
public:


typedef Geometry<TPointType> BaseType;


typedef Line2D4<TPointType> EdgeType;


KRATOS_CLASS_POINTER_DEFINITION(Triangle2D10);


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


Triangle2D10(const PointType& Point01, const PointType& Point02, const PointType& Point03,
const PointType& Point04, const PointType& Point05, const PointType& Point06,
const PointType& Point07, const PointType& Point08, const PointType& Point09,
const PointType& Point10) : BaseType(PointsArrayType(), &msGeometryData)
{
auto& r_points = this->Points();
r_points.push_back(typename PointType::Pointer(new PointType(Point01)));
r_points.push_back(typename PointType::Pointer(new PointType(Point02)));
r_points.push_back(typename PointType::Pointer(new PointType(Point03)));
r_points.push_back(typename PointType::Pointer(new PointType(Point04)));
r_points.push_back(typename PointType::Pointer(new PointType(Point05)));
r_points.push_back(typename PointType::Pointer(new PointType(Point06)));
r_points.push_back(typename PointType::Pointer(new PointType(Point07)));
r_points.push_back(typename PointType::Pointer(new PointType(Point08)));
r_points.push_back(typename PointType::Pointer(new PointType(Point09)));
r_points.push_back(typename PointType::Pointer(new PointType(Point10)));
}

Triangle2D10(typename PointType::Pointer pPoint01, typename PointType::Pointer pPoint02,
typename PointType::Pointer pPoint03, typename PointType::Pointer pPoint04,
typename PointType::Pointer pPoint05, typename PointType::Pointer pPoint06,
typename PointType::Pointer pPoint07, typename PointType::Pointer pPoint08,
typename PointType::Pointer pPoint09, typename PointType::Pointer pPoint10)
: BaseType(PointsArrayType(), &msGeometryData)
{
auto& r_points = this->Points();
r_points.push_back(pPoint01);
r_points.push_back(pPoint02);
r_points.push_back(pPoint03);
r_points.push_back(pPoint04);
r_points.push_back(pPoint05);
r_points.push_back(pPoint06);
r_points.push_back(pPoint07);
r_points.push_back(pPoint08);
r_points.push_back(pPoint09);
r_points.push_back(pPoint10);
}

Triangle2D10(const PointsArrayType& ThisPoints) : BaseType(ThisPoints, &msGeometryData)
{
if (this->PointsNumber() != 10)
{
KRATOS_ERROR << "Invalid points number. Expected 10, given " << this->PointsNumber()
<< std::endl;
}
}

explicit Triangle2D10(const IndexType GeometryId, const PointsArrayType& rThisPoints)
: BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 10) << "Invalid points number. Expected 10, given "
<< this->PointsNumber() << std::endl;
}

explicit Triangle2D10(const std::string& rGeometryName, const PointsArrayType& rThisPoints)
: BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 10) << "Invalid points number. Expected 10, given "
<< this->PointsNumber() << std::endl;
}


Triangle2D10(Triangle2D10 const& rOther) : BaseType(rOther)
{
}


template<class TOtherPointType> Triangle2D10(Triangle2D10<TOtherPointType> const& rOther)
: BaseType(rOther)
{
}


~Triangle2D10() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Triangle;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Triangle2D10;
}



Triangle2D10& operator=(const Triangle2D10& rOther)
{
BaseType::operator=(rOther);
return *this;
}


template<class TOtherPointType>
Triangle2D10& operator=(Triangle2D10<TOtherPointType> const& rOther)
{
BaseType::operator=(rOther);
return *this;
}



typename BaseType::Pointer Create(PointsArrayType const& rThisPoints) const override
{
return typename BaseType::Pointer(new Triangle2D10(rThisPoints));
}


typename BaseType::Pointer Create(const IndexType NewGeometryId,
PointsArrayType const& rThisPoints) const override
{
return typename BaseType::Pointer(new Triangle2D10(NewGeometryId, rThisPoints));
}


typename BaseType::Pointer Create(const BaseType& rGeometry) const override
{
auto p_geometry = typename BaseType::Pointer(new Triangle2D10(rGeometry.Points()));
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


typename BaseType::Pointer Create(const IndexType NewGeometryId, const BaseType& rGeometry) const override
{
auto p_geometry = typename BaseType::Pointer(new Triangle2D10(NewGeometryId, rGeometry.Points()));
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Matrix& PointsLocalCoordinates(Matrix& rResult) const override
{
rResult.resize(10, 2, false);
const double oneThird = 1.0 / 3.0;
const double twoThird = 2.0 / 3.0;
rResult(0, 0) = 0.0;
rResult(0, 1) = 0.0;
rResult(1, 0) = 1.0;
rResult(1, 1) = 0.0;
rResult(2, 0) = 0.0;
rResult(2, 1) = 1.0;
rResult(3, 0) = oneThird;
rResult(3, 1) = 0.0;
rResult(4, 0) = twoThird;
rResult(4, 1) = 0.0;
rResult(5, 0) = twoThird;
rResult(5, 1) = oneThird;
rResult(6, 0) = oneThird;
rResult(6, 1) = twoThird;
rResult(7, 0) = 0.0;
rResult(7, 1) = twoThird;
rResult(8, 0) = 0.0;
rResult(8, 1) = oneThird;
rResult(9, 0) = oneThird;
rResult(9, 1) = oneThird;
return rResult;
}



double Length() const override
{
return std::sqrt(std::abs(Area()));
}


double Area() const override
{
Vector temp;
this->DeterminantOfJacobian(temp, msGeometryData.DefaultIntegrationMethod());
const IntegrationPointsArrayType& integration_points = this->IntegrationPoints(msGeometryData.DefaultIntegrationMethod());
double area = 0.00;
for (unsigned int i = 0; i < integration_points.size(); ++i)
{
area += temp[i] * integration_points[i].Weight();
}
return area;
}


double DomainSize() const override
{
return Area();
}


bool IsInside(const CoordinatesArrayType& rPoint, CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()) const override
{
this->PointLocalCoordinates(rResult, rPoint);
if ((rResult[0] >= (0.0 - Tolerance)) && (rResult[0] <= (1.0 + Tolerance))) {
if ((rResult[1] >= (0.0 - Tolerance)) && (rResult[1] <= (1.0 + Tolerance))) {
if ((rResult[0] + rResult[1]) <= (1.0 + Tolerance)) {
return true;
}
}
}
return false;
}



Vector& ShapeFunctionsValues(Vector& rResult, const CoordinatesArrayType& rCoordinates) const override
{
if (rResult.size() != 10) rResult.resize(10, false);
const double xi = rCoordinates[0];
const double et = rCoordinates[1];
const double zt = 1.0 - xi - et;
rResult[0] = zt * (3.0 * zt - 1.0) * (3.0 * zt - 2.0) * 0.5;
rResult[1] = xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) * 0.5;
rResult[2] = et * (3.0 * et - 1.0) * (3.0 * et - 2.0) * 0.5;
rResult[3] = xi * zt * (3.0 * zt - 1.0) * 4.5;
rResult[4] = xi * zt * (3.0 * xi - 1.0) * 4.5;
rResult[5] = xi * et * (3.0 * xi - 1.0) * 4.5;
rResult[6] = xi * et * (3.0 * et - 1.0) * 4.5;
rResult[7] = et * zt * (3.0 * et - 1.0) * 4.5;
rResult[8] = et * zt * (3.0 * zt - 1.0) * 4.5;
rResult[9] = xi * et * zt * 27.0;
return rResult;
}


double ShapeFunctionValue(IndexType ShapeFunctionIndex, const CoordinatesArrayType& rPoint) const override
{
const double xi = rPoint[0];
const double et = rPoint[1];
const double zt = 1.0 - xi - et;
double shape = 0.0;
switch (ShapeFunctionIndex)
{
case 0:
shape = zt * (3.0 * zt - 1.0) * (3.0 * zt - 2.0) * 0.5;
break;
case 1:
shape = xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) * 0.5;
break;
case 2:
shape = et * (3.0 * et - 1.0) * (3.0 * et - 2.0) * 0.5;
break;
case 3:
shape = xi * zt * (3.0 * zt - 1.0) * 4.5;
break;
case 4:
shape = xi * zt * (3.0 * xi - 1.0) * 4.5;
break;
case 5:
shape = xi * et * (3.0 * xi - 1.0) * 4.5;
break;
case 6:
shape = xi * et * (3.0 * et - 1.0) * 4.5;
break;
case 7:
shape = et * zt * (3.0 * et - 1.0) * 4.5;
break;
case 8:
shape = et * zt * (3.0 * zt - 1.0) * 4.5;
break;
case 9:
shape = xi * et * zt * 27.0;
break;
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
break;
}
return shape;
}



std::string Info() const override
{
return "2 dimensional triangle with ten nodes in 2D space";
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "2 dimensional triangle with ten nodes in 2D space";
}


void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
BaseType::PrintData(rOStream);
std::cout << std::endl;
Matrix jacobian;
this->Jacobian(jacobian, PointType());
rOStream << "    Jacobian in the origin\t : " << jacobian;
}



SizeType EdgesNumber() const override
{
return 3;
}


GeometriesArrayType GenerateEdges() const override
{
GeometriesArrayType edges = GeometriesArrayType();
edges.push_back(Kratos::make_shared<EdgeType>(this->pGetPoint(0), this->pGetPoint(1), this->pGetPoint(3), this->pGetPoint(4)));
edges.push_back(Kratos::make_shared<EdgeType>(this->pGetPoint(1), this->pGetPoint(2), this->pGetPoint(5), this->pGetPoint(6)));
edges.push_back(Kratos::make_shared<EdgeType>(this->pGetPoint(2), this->pGetPoint(0), this->pGetPoint(7), this->pGetPoint(8)));
return edges;
}

SizeType FacesNumber() const override
{
return 3;
}

void NumberNodesInFaces(DenseVector<unsigned int>& NumberNodesInFaces) const override
{
if (NumberNodesInFaces.size() != 3) NumberNodesInFaces.resize(3, false);
NumberNodesInFaces[0] = 4;
NumberNodesInFaces[1] = 4;
NumberNodesInFaces[2] = 4;
}

void NodesInFaces(DenseMatrix<unsigned int>& NodesInFaces) const override
{
if (NodesInFaces.size1() != 5 || NodesInFaces.size2() != 3)
NodesInFaces.resize(5, 3, false);
NodesInFaces(0, 0) = 0;
NodesInFaces(1, 0) = 1;
NodesInFaces(2, 0) = 5;
NodesInFaces(3, 0) = 6;
NodesInFaces(4, 0) = 2;
NodesInFaces(0, 1) = 1;
NodesInFaces(1, 1) = 2;
NodesInFaces(2, 1) = 7;
NodesInFaces(3, 1) = 8;
NodesInFaces(4, 1) = 0;
NodesInFaces(0, 2) = 2;
NodesInFaces(1, 2) = 0;
NodesInFaces(2, 2) = 3;
NodesInFaces(3, 2) = 4;
NodesInFaces(4, 2) = 1;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients(IntegrationMethod ThisMethod)
{
ShapeFunctionsGradientsType localGradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients(ThisMethod);
const int integration_points_number = msGeometryData.IntegrationPointsNumber(ThisMethod);
ShapeFunctionsGradientsType Result(integration_points_number);
for (int pnt = 0; pnt < integration_points_number; ++pnt)
{
Result[pnt] = localGradients[pnt];
}
return Result;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients()
{
IntegrationMethod ThisMethod = msGeometryData.DefaultIntegrationMethod();
ShapeFunctionsGradientsType localGradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients(ThisMethod);
const int integration_points_number = msGeometryData.IntegrationPointsNumber(ThisMethod);
ShapeFunctionsGradientsType Result(integration_points_number);
for (int pnt = 0; pnt < integration_points_number; ++pnt)
{
Result[pnt] = localGradients[pnt];
}
return Result;
}


Matrix& ShapeFunctionsLocalGradients(Matrix& rResult, const CoordinatesArrayType& rPoint) const override
{
rResult.resize(10, 2, false);
const double xi = rPoint[0];
const double et = rPoint[1];
const double zt = 1.0 - xi - et;
noalias(rResult) = ZeroMatrix(10, 2);
rResult(0, 0) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
rResult(0, 1) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
rResult(1, 0) = 4.5 * xi * (3.0 * xi - 2.0) + 1.0;
rResult(1, 1) = 0.0;
rResult(2, 0) = 0.0;
rResult(2, 1) = 4.5 * et * (3.0 * et - 2.0) + 1.0;
rResult(3, 0) = 4.5 * (zt * (3.0 * zt - 1.0) - xi * (6.0 * zt - 1.0));
rResult(3, 1) = -4.5 * xi * (6.0 * zt - 1.0);
rResult(4, 0) = 4.5 * (zt * (6.0 * xi - 1.0) - xi * (3.0 * xi - 1.0));
rResult(4, 1) = -4.5 * xi * (3.0 * xi - 1.0);
rResult(5, 0) = 4.5 * et * (6.0 * xi - 1.0);
rResult(5, 1) = 4.5 * xi * (3.0 * xi - 1.0);
rResult(6, 0) = 4.5 * et * (3.0 * et - 1.0);
rResult(6, 1) = 4.5 * xi * (6.0 * et - 1.0);
rResult(7, 0) = -4.5 * et * (3.0 * et - 1.0);
rResult(7, 1) = 4.5 * (zt * (6.0 * et - 1.0) - et * (3.0 * et - 1.0));
rResult(8, 0) = -4.5 * et * (6.0 * zt - 1.0);
rResult(8, 1) = 4.5 * (zt * (3.0 * zt - 1.0) - et * (6.0 * zt - 1.0));
rResult(9, 0) = 27.0 * et * (zt - xi);
rResult(9, 1) = 27.0 * xi * (zt - et);
return rResult;
}


virtual Matrix& ShapeFunctionsGradients(Matrix& rResult, const CoordinatesArrayType& rPoint)
{
rResult.resize(10, 2, false);
const double xi = rPoint[0];
const double et = rPoint[1];
const double zt = 1.0 - xi - et;
noalias(rResult) = ZeroMatrix(10, 2);
rResult(0, 0) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
rResult(0, 1) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
rResult(1, 0) = 4.5 * xi * (3.0 * xi - 2.0) + 1.0;
rResult(1, 1) = 0.0;
rResult(2, 0) = 0.0;
rResult(2, 1) = 4.5 * et * (3.0 * et - 2.0) + 1.0;
rResult(3, 0) = 4.5 * (zt * (3.0 * zt - 1.0) - xi * (6.0 * zt - 1.0));
rResult(3, 1) = -4.5 * xi * (6.0 * zt - 1.0);
rResult(4, 0) = 4.5 * (zt * (6.0 * xi - 1.0) - xi * (3.0 * xi - 1.0));
rResult(4, 1) = -4.5 * xi * (3.0 * xi - 1.0);
rResult(5, 0) = 4.5 * et * (6.0 * xi - 1.0);
rResult(5, 1) = 4.5 * xi * (3.0 * xi - 1.0);
rResult(6, 0) = 4.5 * et * (3.0 * et - 1.0);
rResult(6, 1) = 4.5 * xi * (6.0 * et - 1.0);
rResult(7, 0) = -4.5 * et * (3.0 * et - 1.0);
rResult(7, 1) = 4.5 * (zt * (6.0 * et - 1.0) - et * (3.0 * et - 1.0));
rResult(8, 0) = -4.5 * et * (6.0 * zt - 1.0);
rResult(8, 1) = 4.5 * (zt * (3.0 * zt - 1.0) - et * (6.0 * zt - 1.0));
rResult(9, 0) = 27.0 * et * (zt - xi);
rResult(9, 1) = 27.0 * xi * (zt - et);
return rResult;
}


ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives(
ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint) const override
{
if (rResult.size() != this->PointsNumber())
{
ShapeFunctionsGradientsType temp(this->PointsNumber());
rResult.swap(temp);
}
rResult[0].resize(2, 2, false);
rResult[1].resize(2, 2, false);
rResult[2].resize(2, 2, false);
rResult[3].resize(2, 2, false);
rResult[4].resize(2, 2, false);
rResult[5].resize(2, 2, false);
rResult[6].resize(2, 2, false);
rResult[7].resize(2, 2, false);
rResult[8].resize(2, 2, false);
rResult[9].resize(2, 2, false);
const double xi = rPoint[0];
const double et = rPoint[1];
const double zt = 1.0 - xi - et;
rResult[0](0, 0) = 9.0 * (3.0 * zt - 1.0);
rResult[0](0, 1) = 9.0 * (3.0 * zt - 1.0);
rResult[0](1, 0) = 9.0 * (3.0 * zt - 1.0);
rResult[0](1, 1) = 9.0 * (3.0 * zt - 1.0);
rResult[1](0, 0) = 9.0 * (3.0 * xi - 1.0);
rResult[1](0, 1) = 0.0;
rResult[1](1, 0) = 0.0;
rResult[1](1, 1) = 0.0;
rResult[2](0, 0) = 0.0;
rResult[2](0, 1) = 0.0;
rResult[2](1, 0) = 0.0;
rResult[2](1, 1) = 9.0 * (3.0 * et - 1.0);
rResult[3](0, 0) = 9.0 * (3.0 * xi - 6.0 * zt + 1.0);
rResult[3](0, 1) = 4.5 * (6.0 * xi - 6.0 * zt + 1.0);
rResult[3](1, 0) = 4.5 * (6.0 * xi - 6.0 * zt + 1.0);
rResult[3](1, 1) = 27.0 * xi;
rResult[4](0, 0) = 9.0 * (3.0 * zt - 6.0 * xi + 1.0);
rResult[4](0, 1) = -4.5 * (6.0 * xi - 1.0);
rResult[4](1, 0) = -4.5 * (6.0 * xi - 1.0);
rResult[4](1, 1) = 0.0;
rResult[5](0, 0) = 27.0 * et;
rResult[5](0, 1) = 4.5 * (6.0 * xi - 1.0);
rResult[5](1, 0) = 4.5 * (6.0 * xi - 1.0);
rResult[5](1, 1) = 0.0;
rResult[6](0, 0) = 0.0;
rResult[6](0, 1) = 4.5 * (6.0 * et - 1.0);
rResult[6](1, 0) = 4.5 * (6.0 * et - 1.0);
rResult[6](1, 1) = 27.0 * xi;
rResult[7](0, 0) = 0.0;
rResult[7](0, 1) = 4.5 * (6.0 * et - 1.0);
rResult[7](1, 0) = 4.5 * (6.0 * et - 1.0);
rResult[7](1, 1) = 9.0 * (3.0 * zt - 6.0 * et + 1.0);
rResult[8](0, 0) = 27.0 * et;
rResult[8](0, 1) = 4.5 * (6.0 * et - 6.0 * zt + 1.0);
rResult[8](1, 0) = 4.5 * (6.0 * et - 6.0 * zt + 1.0);
rResult[8](1, 1) = 9.0 * (3.0 * et - 6.0 * zt + 1.0);
rResult[9](0, 0) = -54.0 * et;
rResult[9](0, 1) = -27.0 * (xi - zt + et);
rResult[9](1, 0) = -27.0 * (xi - zt + et);
rResult[9](1, 1) = -54.0 * xi;
return rResult;
}


ShapeFunctionsThirdDerivativesType& ShapeFunctionsThirdDerivatives(
ShapeFunctionsThirdDerivativesType& rResult, const CoordinatesArrayType& rPoint) const override
{
if (rResult.size() != this->PointsNumber())
{
ShapeFunctionsThirdDerivativesType temp(this->PointsNumber());
rResult.swap(temp);
}
for (IndexType i = 0; i < rResult.size(); ++i)
{
DenseVector<Matrix> temp(this->PointsNumber());
rResult[i].swap(temp);
}
rResult[0][0].resize(2, 2, false);
rResult[0][1].resize(2, 2, false);
rResult[1][0].resize(2, 2, false);
rResult[1][1].resize(2, 2, false);
rResult[2][0].resize(2, 2, false);
rResult[2][1].resize(2, 2, false);
rResult[3][0].resize(2, 2, false);
rResult[3][1].resize(2, 2, false);
rResult[4][0].resize(2, 2, false);
rResult[4][1].resize(2, 2, false);
rResult[5][0].resize(2, 2, false);
rResult[5][1].resize(2, 2, false);
rResult[6][0].resize(2, 2, false);
rResult[6][1].resize(2, 2, false);
rResult[7][0].resize(2, 2, false);
rResult[7][1].resize(2, 2, false);
rResult[8][0].resize(2, 2, false);
rResult[8][1].resize(2, 2, false);
rResult[9][0].resize(2, 2, false);
rResult[9][1].resize(2, 2, false);
rResult[0][0](0, 0) = -27.0;
rResult[0][0](0, 1) = -27.0;
rResult[0][0](1, 0) = -27.0;
rResult[0][0](1, 1) = -27.0;
rResult[0][1](0, 0) = -27.0;
rResult[0][1](0, 1) = -27.0;
rResult[0][1](1, 0) = -27.0;
rResult[0][1](1, 1) = -27.0;
rResult[1][0](0, 0) = 27.0;
rResult[1][0](0, 1) = 0.0;
rResult[1][0](1, 0) = 0.0;
rResult[1][0](1, 1) = 0.0;
rResult[1][1](0, 0) = 0.0;
rResult[1][1](0, 1) = 0.0;
rResult[1][1](1, 0) = 0.0;
rResult[1][1](1, 1) = 0.0;
rResult[2][0](0, 0) = 0.0;
rResult[2][0](0, 1) = 0.0;
rResult[2][0](1, 0) = 0.0;
rResult[2][0](1, 1) = 0.0;
rResult[2][1](0, 0) = 0.0;
rResult[2][1](0, 1) = 0.0;
rResult[2][1](1, 0) = 0.0;
rResult[2][1](1, 1) = 27.0;
rResult[3][0](0, 0) = 81.0;
rResult[3][0](0, 1) = 54.0;
rResult[3][0](1, 0) = 54.0;
rResult[3][0](1, 1) = 27.0;
rResult[3][1](0, 0) = 54.0;
rResult[3][1](0, 1) = 27.0;
rResult[3][1](1, 0) = 27.0;
rResult[3][1](1, 1) = 0.0;
rResult[4][0](0, 0) = -81.0;
rResult[4][0](0, 1) = -27.0;
rResult[4][0](1, 0) = -27.0;
rResult[4][0](1, 1) = 0.0;
rResult[4][1](0, 0) = -27.0;
rResult[4][1](0, 1) = 0.0;
rResult[4][1](1, 0) = 0.0;
rResult[4][1](1, 1) = 0.0;
rResult[5][0](0, 0) = 0.0;
rResult[5][0](0, 1) = 27.0;
rResult[5][0](1, 0) = 27.0;
rResult[5][0](1, 1) = 0.0;
rResult[5][1](0, 0) = 27.0;
rResult[5][1](0, 1) = 0.0;
rResult[5][1](1, 0) = 0.0;
rResult[5][1](1, 1) = 0.0;
rResult[6][0](0, 0) = 0.0;
rResult[6][0](0, 1) = 0.0;
rResult[6][0](1, 0) = 0.0;
rResult[6][0](1, 1) = 27.0;
rResult[6][1](0, 0) = 0.0;
rResult[6][1](0, 1) = 27.0;
rResult[6][1](1, 0) = 27.0;
rResult[6][1](1, 1) = 0.0;
rResult[7][0](0, 0) = 0.0;
rResult[7][0](0, 1) = 0.0;
rResult[7][0](1, 0) = 0.0;
rResult[7][0](1, 1) = -27.0;
rResult[7][1](0, 0) = 0.0;
rResult[7][1](0, 1) = -27.0;
rResult[7][1](1, 0) = -27.0;
rResult[7][1](1, 1) = -81.0;
rResult[8][0](0, 0) = 0.0;
rResult[8][0](0, 1) = 27.0;
rResult[8][0](1, 0) = 27.0;
rResult[8][0](1, 1) = 54.0;
rResult[8][1](0, 0) = 27.0;
rResult[8][1](0, 1) = 54.0;
rResult[8][1](1, 0) = 54.0;
rResult[8][1](1, 1) = 81.0;
rResult[9][0](0, 0) = 0.0;
rResult[9][0](0, 1) = -54.0;
rResult[9][0](1, 0) = -54.0;
rResult[9][0](1, 1) = -54.0;
rResult[9][1](0, 0) = -54.0;
rResult[9][1](0, 1) = -54.0;
rResult[9][1](1, 0) = -54.0;
rResult[9][1](1, 1) = 0.0;
return rResult;
}



protected:


private:

static const GeometryData msGeometryData;

static const GeometryDimension msGeometryDimension;


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, BaseType);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, BaseType);
}

Triangle2D10() : BaseType(PointsArrayType(), &msGeometryData) {}





static Matrix CalculateShapeFunctionsIntegrationPointsValues(typename BaseType::IntegrationMethod ThisMethod)
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
const int points_number = 10;
Matrix shape_function_values(integration_points_number, points_number);
for (int pnt = 0; pnt < integration_points_number; ++pnt)
{
const double xi = integration_points[pnt].X();
const double et = integration_points[pnt].Y();
const double zt = 1.0 - xi - et;
shape_function_values(pnt, 0) = zt * (3.0 * zt - 1.0) * (3.0 * zt - 2.0) * 0.5;
shape_function_values(pnt, 1) = xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) * 0.5;
shape_function_values(pnt, 2) = et * (3.0 * et - 1.0) * (3.0 * et - 2.0) * 0.5;
shape_function_values(pnt, 3) = xi * zt * (3.0 * zt - 1.0) * 4.5;
shape_function_values(pnt, 4) = xi * zt * (3.0 * xi - 1.0) * 4.5;
shape_function_values(pnt, 5) = xi * et * (3.0 * xi - 1.0) * 4.5;
shape_function_values(pnt, 6) = xi * et * (3.0 * et - 1.0) * 4.5;
shape_function_values(pnt, 7) = et * zt * (3.0 * et - 1.0) * 4.5;
shape_function_values(pnt, 8) = et * zt * (3.0 * zt - 1.0) * 4.5;
shape_function_values(pnt, 9) = xi * et * zt * 27.0;
}
return shape_function_values;
}


static ShapeFunctionsGradientsType
CalculateShapeFunctionsIntegrationPointsLocalGradients(typename BaseType::IntegrationMethod ThisMethod)
{
IntegrationPointsContainerType all_integration_points = AllIntegrationPoints();
IntegrationPointsArrayType integration_points = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = integration_points.size();
ShapeFunctionsGradientsType d_shape_f_values(integration_points_number);
for (int pnt = 0; pnt < integration_points_number; ++pnt)
{
Matrix result(10, 2);
const double xi = integration_points[pnt].X();
const double et = integration_points[pnt].Y();
const double zt = 1.0 - xi - et;
noalias(result) = ZeroMatrix(10, 2);
result(0, 0) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
result(0, 1) = -4.5 * zt * (3.0 * zt - 2.0) - 1.0;
result(1, 0) = 4.5 * xi * (3.0 * xi - 2.0) + 1.0;
result(1, 1) = 0.0;
result(2, 0) = 0.0;
result(2, 1) = 4.5 * et * (3.0 * et - 2.0) + 1.0;
result(3, 0) = 4.5 * (zt * (3.0 * zt - 1.0) - xi * (6.0 * zt - 1.0));
result(3, 1) = -4.5 * xi * (6.0 * zt - 1.0);
result(4, 0) = 4.5 * (zt * (6.0 * xi - 1.0) - xi * (3.0 * xi - 1.0));
result(4, 1) = -4.5 * xi * (3.0 * xi - 1.0);
result(5, 0) = 4.5 * et * (6.0 * xi - 1.0);
result(5, 1) = 4.5 * xi * (3.0 * xi - 1.0);
result(6, 0) = 4.5 * et * (3.0 * et - 1.0);
result(6, 1) = 4.5 * xi * (6.0 * et - 1.0);
result(7, 0) = -4.5 * et * (3.0 * et - 1.0);
result(7, 1) = 4.5 * (zt * (6.0 * et - 1.0) - et * (3.0 * et - 1.0));
result(8, 0) = -4.5 * et * (6.0 * zt - 1.0);
result(8, 1) = 4.5 * (zt * (3.0 * zt - 1.0) - et * (6.0 * zt - 1.0));
result(9, 0) = 27.0 * et * (zt - xi);
result(9, 1) = 27.0 * xi * (zt - et);
d_shape_f_values[pnt] = result;
}
return d_shape_f_values;
}

static const IntegrationPointsContainerType AllIntegrationPoints()
{
IntegrationPointsContainerType integration_points =
{
{
Quadrature<TriangleGaussLegendreIntegrationPoints1, 2, IntegrationPoint<3>>::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints2, 2, IntegrationPoint<3>>::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints3, 2, IntegrationPoint<3>>::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints4, 2, IntegrationPoint<3>>::GenerateIntegrationPoints(),
Quadrature<TriangleGaussLegendreIntegrationPoints5, 2, IntegrationPoint<3>>::GenerateIntegrationPoints()
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values =
{
{
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_1),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_2),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_3),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_4),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_5),
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients =
{
{
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_1),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_2),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_3),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_4),
Triangle2D10<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_5),
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Triangle2D10;


}; 




template<class TPointType>
inline std::istream& operator >> (std::istream& rIStream, Triangle2D10<TPointType>& rThis);


template<class TPointType>
inline std::ostream& operator << (std::ostream& rOStream, const Triangle2D10<TPointType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);
return rOStream;
}


template<class TPointType> const
GeometryData Triangle2D10<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_4,
Triangle2D10<TPointType>::AllIntegrationPoints(),
Triangle2D10<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients());

template<class TPointType>
const GeometryDimension Triangle2D10<TPointType>::msGeometryDimension(2, 2);

}