
#pragma once



#include "geometries/geometry.h"
#include "integration/line_gauss_legendre_integration_points.h"
#include "utilities/integration_utilities.h"

namespace Kratos
{







template<class TPointType>
class Line2D4 : public Geometry<TPointType>
{
public:

typedef Geometry<TPointType> BaseType;

KRATOS_CLASS_POINTER_DEFINITION(Line2D4);


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


Line2D4(const PointType& Point01, const PointType& Point02, const PointType& Point03,
const PointType& Point04) : BaseType(PointsArrayType(), &msGeometryData)
{
BaseType::Points().push_back(typename PointType::Pointer(new PointType(Point01)));
BaseType::Points().push_back(typename PointType::Pointer(new PointType(Point02)));
BaseType::Points().push_back(typename PointType::Pointer(new PointType(Point03)));
BaseType::Points().push_back(typename PointType::Pointer(new PointType(Point04)));
}

Line2D4(typename PointType::Pointer pPoint01, typename PointType::Pointer pPoint02,
typename PointType::Pointer pPoint03, typename PointType::Pointer pPoint04)
: BaseType(PointsArrayType(), &msGeometryData)
{
BaseType::Points().push_back(pPoint01);
BaseType::Points().push_back(pPoint02);
BaseType::Points().push_back(pPoint03);
BaseType::Points().push_back(pPoint04);
}

explicit Line2D4(const PointsArrayType& rThisPoints) : BaseType(rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(BaseType::PointsNumber() != 4) << "Invalid points number. Expected 4, given "
<< BaseType::PointsNumber() << std::endl;
}

explicit Line2D4(const IndexType GeometryId, const PointsArrayType& rThisPoints)
: BaseType(GeometryId, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 4) << "Invalid points number. Expected 4, given "
<< this->PointsNumber() << std::endl;
}

explicit Line2D4(const std::string& rGeometryName, const PointsArrayType& rThisPoints)
: BaseType(rGeometryName, rThisPoints, &msGeometryData)
{
KRATOS_ERROR_IF(this->PointsNumber() != 4) << "Invalid points number. Expected 4, given "
<< this->PointsNumber() << std::endl;
}


Line2D4(Line2D4 const& rOther) : BaseType(rOther)
{
}


template<class TOtherPointType> explicit Line2D4(Line2D4<TOtherPointType> const& rOther)
: BaseType(rOther)
{
}

~Line2D4() override {}

GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Linear;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Line2D4;
}



Line2D4& operator=(const Line2D4& rOther)
{
BaseType::operator=(rOther);
return *this;
}


template<class TOtherPointType>
Line2D4& operator=(Line2D4<TOtherPointType> const& rOther)
{
BaseType::operator=(rOther);
return *this;
}



typename BaseType::Pointer Create(const IndexType NewGeometryId, PointsArrayType const& rThisPoints)
const override
{
return typename BaseType::Pointer(new Line2D4(NewGeometryId, rThisPoints));
}


typename BaseType::Pointer Create(const IndexType NewGeometryId, const BaseType& rGeometry) const override
{
auto p_geometry = typename BaseType::Pointer(new Line2D4(NewGeometryId, rGeometry.Points()));
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Vector& LumpingFactors(Vector& rResult, const typename BaseType::LumpingMethods LumpingMethod
= BaseType::LumpingMethods::ROW_SUM) const override
{
if (rResult.size() != 4) rResult.resize(4, false);
rResult[0] = 0.125;
rResult[1] = 0.125;
rResult[2] = 0.375;
rResult[3] = 0.375;
return rResult;
}



double Length() const override
{
Vector temp;
const IntegrationMethod integration_method = IntegrationUtilities::GetIntegrationMethodForExactMassMatrixEvaluation(*this);
this->DeterminantOfJacobian(temp, integration_method);
const IntegrationPointsArrayType& r_integration_points = this->IntegrationPoints(integration_method);
double length = 0.0;

for (std::size_t i = 0; i < r_integration_points.size(); ++i) {
length += temp[i] * r_integration_points[i].Weight();
}
return length;
}


double Area() const override
{
return Length();
}


double DomainSize() const override
{
return Length();
}


bool IsInside(const CoordinatesArrayType& rPoint, CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()) const override
{
PointLocalCoordinates(rResult, rPoint);
if (std::abs(rResult[0]) <= (1.0 + Tolerance)) {
return true;
}
return false;
}


CoordinatesArrayType& PointLocalCoordinates(CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint) const override
{
BoundedMatrix<double, 3, 4> X;
BoundedMatrix<double, 3, 1> DN;
for (IndexType i = 0; i < this->size(); ++i)
{
const auto& r_node = this->GetPoint(i);
X(0, i) = r_node.X();
X(1, i) = r_node.Y();
X(2, i) = r_node.Z();
}

static constexpr double MaxNormPointLocalCoordinates = 300.0;
static constexpr std::size_t MaxIteratioNumberPointLocalCoordinates = 500;
static constexpr double MaxTolerancePointLocalCoordinates = 1.0e-8;

Matrix J = ZeroMatrix(1, 1);
Matrix invJ = ZeroMatrix(1, 1);

if (rResult.size() != 3) rResult.resize(3, false);
noalias(rResult) = ZeroVector(3);
double delta_xi = 0.0;
const array_1d<double, 3> zero_array = ZeroVector(3);
array_1d<double, 3> current_global_coords;
array_1d<double, 1> res;

for (IndexType k = 0; k < MaxIteratioNumberPointLocalCoordinates; ++k) {
noalias(current_global_coords) = zero_array;
this->GlobalCoordinates(current_global_coords, rResult);

noalias(current_global_coords) = rPoint - current_global_coords;

Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(shape_functions_gradients, rResult);
noalias(DN) = prod(X, shape_functions_gradients);
noalias(J) = prod(trans(DN), DN);
noalias(res) = prod(trans(DN), current_global_coords);

invJ(0, 0) = 1.0 / J(0, 0);

delta_xi = invJ(0, 0) * res[0];

rResult[0] += delta_xi;

if (delta_xi > MaxNormPointLocalCoordinates) {
KRATOS_WARNING_IF("Line2D4", k > 0) << "detJ =\t" << J(0, 0) << " DeltaX =\t" << delta_xi << " stopping calculation. Iteration:\t" << k << std::endl;
break;
}

if (delta_xi < MaxTolerancePointLocalCoordinates)
break;
}

return rResult;
}



JacobiansType& Jacobian(JacobiansType& rResult, IntegrationMethod ThisMethod) const override
{
const ShapeFunctionsGradientsType shape_functions_gradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients(ThisMethod);
const std::size_t number_of_integration_points = this->IntegrationPointsNumber(ThisMethod);
if (rResult.size() != number_of_integration_points) {
JacobiansType temp(number_of_integration_points);
rResult.swap(temp);
}
for (std::size_t pnt = 0; pnt < number_of_integration_points; ++pnt) {
noalias(rResult[pnt]) = ZeroMatrix(2, 1);
for (std::size_t i = 0; i < this->PointsNumber(); ++i) {
const auto& r_node = this->GetPoint(i);
rResult[pnt](0, 0) += r_node.X() * shape_functions_gradients[pnt](i, 0);
rResult[pnt](1, 0) += r_node.Y() * shape_functions_gradients[pnt](i, 0);
}
} 
return rResult;
}


JacobiansType& Jacobian(JacobiansType& rResult, IntegrationMethod ThisMethod,
Matrix& rDeltaPosition) const override
{
ShapeFunctionsGradientsType shape_functions_gradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients(ThisMethod);
const std::size_t number_of_integration_points = this->IntegrationPointsNumber(ThisMethod);
Matrix shape_functions_values = CalculateShapeFunctionsIntegrationPointsValues(ThisMethod);
if (rResult.size() != number_of_integration_points) {
JacobiansType temp(number_of_integration_points);
rResult.swap(temp);
}
for (std::size_t pnt = 0; pnt < number_of_integration_points; ++pnt) {
noalias(rResult[pnt]) = ZeroMatrix(2, 1);
for (std::size_t i = 0; i < this->PointsNumber(); ++i) {
const auto& r_node = this->GetPoint(i);
rResult[pnt](0, 0) += (r_node.X() - rDeltaPosition(i, 0)) * shape_functions_gradients[pnt](i, 0);
rResult[pnt](1, 0) += (r_node.Y() - rDeltaPosition(i, 1)) * shape_functions_gradients[pnt](i, 0);
}
}
return rResult;
}


Matrix& Jacobian(Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod)
const override
{
rResult.resize(2, 1, false);
noalias(rResult) = ZeroMatrix(2, 1);
ShapeFunctionsGradientsType shape_functions_gradients
= CalculateShapeFunctionsIntegrationPointsLocalGradients(ThisMethod);
Matrix shape_function_gradient_in_integration_point = shape_functions_gradients(IntegrationPointIndex);
DenseVector<double> ShapeFunctionsValuesInIntegrationPoint = ZeroVector(3);
ShapeFunctionsValuesInIntegrationPoint = row(CalculateShapeFunctionsIntegrationPointsValues(ThisMethod), IntegrationPointIndex);
for (std::size_t i = 0; i < this->PointsNumber(); ++i) {
const auto& r_node = this->GetPoint(i);
rResult(0, 0) += r_node.X() * shape_function_gradient_in_integration_point(i, 0);
rResult(1, 0) += r_node.Y() * shape_function_gradient_in_integration_point(i, 0);
}
return rResult;
}


Matrix& Jacobian(Matrix& rResult, const CoordinatesArrayType& rPoint) const override
{
rResult.resize(2, 1, false);
noalias(rResult) = ZeroMatrix(2, 1);
Matrix shape_functions_gradients;
shape_functions_gradients = ShapeFunctionsLocalGradients(shape_functions_gradients, rPoint);
for (std::size_t i = 0; i < this->PointsNumber(); ++i) {
const auto& r_node = this->GetPoint(i);
rResult(0, 0) += r_node.X() * shape_functions_gradients(i, 0);
rResult(1, 0) += r_node.Y() * shape_functions_gradients(i, 0);
}
return rResult;
}


Vector& DeterminantOfJacobian(Vector& rResult, IntegrationMethod ThisMethod) const override
{
const std::size_t number_of_integration_points = this->IntegrationPointsNumber(ThisMethod);
if (rResult.size() != number_of_integration_points)
rResult.resize(number_of_integration_points, false);
Matrix J(2, 1);
for (std::size_t pnt = 0; pnt < number_of_integration_points; ++pnt) {
this->Jacobian(J, pnt, ThisMethod);
rResult[pnt] = std::sqrt(std::pow(J(0, 0), 2) + std::pow(J(1, 0), 2));
}
return rResult;
}


double DeterminantOfJacobian(IndexType IntegrationPointIndex, IntegrationMethod ThisMethod)
const override
{
Matrix J(2, 1);
this->Jacobian(J, IntegrationPointIndex, ThisMethod);
return std::sqrt(std::pow(J(0, 0), 2) + std::pow(J(1, 0), 2));
}


double DeterminantOfJacobian(const CoordinatesArrayType& rPoint) const override
{
Matrix J(2, 1);
this->Jacobian(J, rPoint);
return std::sqrt(std::pow(J(0, 0), 2) + std::pow(J(1, 0), 2));
}


SizeType EdgesNumber() const override
{
return 2;
}


SizeType FacesNumber() const override
{
return EdgesNumber();
}



Vector& ShapeFunctionsValues(Vector& rResult, const CoordinatesArrayType& rCoordinates) const override
{
if (rResult.size() != 4) rResult.resize(4, false);
const double xi = rCoordinates[0];
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;
rResult[0] = -0.0625 * fx1 * gx3;
rResult[1] = -0.0625 * fx2 * gx3;
rResult[2] =  0.5625 * fx3 * gx1;
rResult[3] =  0.5625 * fx3 * gx2;
return rResult;
}


double ShapeFunctionValue(const IndexType ShapeFunctionIndex, const CoordinatesArrayType& rPoint) const override
{
const double xi = rPoint[0];
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;
double shape = 0.0;
switch (ShapeFunctionIndex)
{
case 0:
shape = -0.0625 * fx1 * gx3;
break;
case 1:
shape = -0.0625 * fx2 * gx3;
break;
case 2:
shape = 0.5625 * fx3 * gx1;
break;
case 3:
shape = 0.5625 * fx3 * gx2;
break;
default:
KRATOS_ERROR << "Wrong index of shape function!" << *this << std::endl;
break;
}
return shape;
}


void ShapeFunctionsIntegrationPointsGradients(ShapeFunctionsGradientsType& rResult,
IntegrationMethod ThisMethod) const override
{
KRATOS_ERROR << "Jacobian is not square" << std::endl;
}

void ShapeFunctionsIntegrationPointsGradients(ShapeFunctionsGradientsType& rResult,
Vector& rDeterminantsOfJacobian, IntegrationMethod ThisMethod) const override
{
KRATOS_ERROR << "Jacobian is not square" << std::endl;
}



std::string Info() const override
{
return "1 dimensional line with 4 nodes in 2D space";
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "1 dimensional line with 4 nodes in 2D space";
}


void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
std::cout << std::endl;
Matrix jacobian;
Jacobian(jacobian, PointType());
rOStream << "    Jacobian\t : " << jacobian;
}


virtual ShapeFunctionsGradientsType ShapeFunctionsLocalGradients(IntegrationMethod& ThisMethod)
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
if (rResult.size1() != 4 || rResult.size2() != 1) rResult.resize(4, 1, false);
noalias(rResult) = ZeroMatrix(4, 1);
const double xi = rPoint[0];
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;
rResult(0, 0) =  0.0625 * (18.0 * xi * fx1 + gx3);
rResult(1, 0) =  0.0625 * (18.0 * xi * fx2 - gx3);
rResult(2, 0) = -0.5625 * (3.0 * fx3 + 2.0 * xi * gx1);
rResult(3, 0) =  0.5625 * (3.0 * fx3 - 2.0 * xi * gx2);
return rResult;
}


Matrix& PointsLocalCoordinates(Matrix& rResult) const override
{
if (rResult.size1() != 4 || rResult.size2() != 1) rResult.resize(4, 1, false);
noalias(rResult) = ZeroMatrix(4, 1);
rResult(0, 0) = -1.0;
rResult(1, 0) =  1.0;
rResult(2, 0) = -1.0 / 3.0;
rResult(3, 0) =  1.0 / 3.0;
return rResult;
}


virtual Matrix& ShapeFunctionsGradients(Matrix& rResult, CoordinatesArrayType& rPoint)
{
if (rResult.size1() != 4 || rResult.size2() != 1) rResult.resize(4, 1, false);
noalias(rResult) = ZeroMatrix(4, 1);

const double xi = rPoint[0];
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;

rResult(0, 0) =  0.0625 * (18.0 * xi * fx1 + gx3);
rResult(1, 0) =  0.0625 * (18.0 * xi * fx2 - gx3);
rResult(2, 0) = -0.5625 * (3.0 * fx3 + 2.0 * xi * gx1);
rResult(3, 0) =  0.5625 * (3.0 * fx3 - 2.0 * xi * gx2);
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

Line2D4() : BaseType(PointsArrayType(), &msGeometryData) {}



static Matrix CalculateShapeFunctionsIntegrationPointsValues(typename BaseType::IntegrationMethod ThisMethod)
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
const int integration_points_number = IntegrationPoints.size();
Matrix shape_function_values(integration_points_number, 4);
for (int pnt = 0; pnt < integration_points_number; ++pnt)
{
const double xi = IntegrationPoints[pnt].X();
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;
shape_function_values(pnt, 0) = -0.0625 * fx1 * gx3;
shape_function_values(pnt, 1) = -0.0625 * fx2 * gx3;
shape_function_values(pnt, 2) =  0.5625 * fx3 * gx1;
shape_function_values(pnt, 3) =  0.5625 * fx3 * gx2;
}
return shape_function_values;
}

static ShapeFunctionsGradientsType CalculateShapeFunctionsIntegrationPointsLocalGradients(
typename BaseType::IntegrationMethod ThisMethod)
{
const IntegrationPointsContainerType& all_integration_points = AllIntegrationPoints();
const IntegrationPointsArrayType& IntegrationPoints = all_integration_points[static_cast<int>(ThisMethod)];
ShapeFunctionsGradientsType DN_De(IntegrationPoints.size());
std::fill(DN_De.begin(), DN_De.end(), Matrix(4, 1));
for (unsigned int pnt = 0; pnt < IntegrationPoints.size(); ++pnt)
{
const double xi = IntegrationPoints[pnt].X();
const double fx1 = 1.0 - xi;
const double fx2 = 1.0 + xi;
const double fx3 = fx1 * fx2;
const double gx1 = 1.0 - 3.0 * xi;
const double gx2 = 1.0 + 3.0 * xi;
const double gx3 = gx1 * gx2;
DN_De[pnt](0, 0) =  0.0625 * (18.0 * xi * fx1 + gx3);
DN_De[pnt](1, 0) =  0.0625 * (18.0 * xi * fx2 - gx3);
DN_De[pnt](2, 0) = -0.5625 * (3.0 * fx3 + 2.0 * xi * gx1);
DN_De[pnt](3, 0) =  0.5625 * (3.0 * fx3 - 2.0 * xi * gx2);
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
Quadrature<LineGaussLegendreIntegrationPoints5, 1, IntegrationPoint<3> >::GenerateIntegrationPoints()
}
};
return integration_points;
}

static const ShapeFunctionsValuesContainerType AllShapeFunctionsValues()
{
ShapeFunctionsValuesContainerType shape_functions_values = {{
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_1),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_2),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_3),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_4),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsValues(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_values;
}

static const ShapeFunctionsLocalGradientsContainerType AllShapeFunctionsLocalGradients()
{
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients = {{
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_1),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_2),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_3),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_4),
Line2D4<TPointType>::CalculateShapeFunctionsIntegrationPointsLocalGradients(GeometryData::IntegrationMethod::GI_GAUSS_5)
}
};
return shape_functions_local_gradients;
}




template<class TOtherPointType> friend class Line2D4;



}; 




template<class TPointType>
inline std::istream& operator >> (std::istream& rIStream, Line2D4<TPointType>& rThis);

template<class TPointType>
inline std::ostream& operator << (std::ostream& rOStream, const Line2D4<TPointType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);
return rOStream;
}


template<class TPointType>
const GeometryData Line2D4<TPointType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_3,
Line2D4<TPointType>::AllIntegrationPoints(),
Line2D4<TPointType>::AllShapeFunctionsValues(),
AllShapeFunctionsLocalGradients());

template<class TPointType>
const GeometryDimension Line2D4<TPointType>::msGeometryDimension(2, 1);

}  