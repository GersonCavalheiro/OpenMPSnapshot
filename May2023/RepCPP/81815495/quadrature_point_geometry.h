
#pragma once



#include "includes/variables.h"
#include "geometries/geometry.h"
#include "geometries/geometry_dimension.h"
#include "utilities/integration_utilities.h"

namespace Kratos
{

template<class TPointType,
int TWorkingSpaceDimension,
int TLocalSpaceDimension = TWorkingSpaceDimension,
int TDimension = TLocalSpaceDimension>
class QuadraturePointGeometry
: public Geometry<TPointType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION( QuadraturePointGeometry );

typedef Geometry<TPointType> BaseType;
typedef Geometry<TPointType> GeometryType;

typedef typename GeometryType::IndexType IndexType;
typedef typename GeometryType::SizeType SizeType;

typedef typename GeometryType::PointsArrayType PointsArrayType;
typedef typename GeometryType::CoordinatesArrayType CoordinatesArrayType;

typedef typename GeometryType::IntegrationPointType IntegrationPointType;
typedef typename GeometryType::IntegrationPointsArrayType IntegrationPointsArrayType;

typedef typename GeometryData::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;

typedef GeometryShapeFunctionContainer<GeometryData::IntegrationMethod> GeometryShapeFunctionContainerType;

typedef typename GeometryType::IntegrationPointsContainerType IntegrationPointsContainerType;
typedef typename GeometryType::ShapeFunctionsValuesContainerType ShapeFunctionsValuesContainerType;
typedef typename GeometryType::ShapeFunctionsLocalGradientsContainerType ShapeFunctionsLocalGradientsContainerType;

using BaseType::Jacobian;
using BaseType::DeterminantOfJacobian;
using BaseType::ShapeFunctionsValues;
using BaseType::ShapeFunctionsLocalGradients;
using BaseType::InverseOfJacobian;


QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const IntegrationPointsContainerType& rIntegrationPoints,
const ShapeFunctionsValuesContainerType& rShapeFunctionValues,
const ShapeFunctionsLocalGradientsContainerType& rShapeFunctionsDerivativesVector)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
rIntegrationPoints,
rShapeFunctionValues,
rShapeFunctionsDerivativesVector)
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const IntegrationPointsContainerType& rIntegrationPoints,
const ShapeFunctionsValuesContainerType& rShapeFunctionValues,
const ShapeFunctionsLocalGradientsContainerType& rShapeFunctionsDerivativesVector,
GeometryType* pGeometryParent)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
rIntegrationPoints,
rShapeFunctionValues,
rShapeFunctionsDerivativesVector)
, mpGeometryParent(pGeometryParent)
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const GeometryShapeFunctionContainerType& ThisGeometryShapeFunctionContainer)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
ThisGeometryShapeFunctionContainer)
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const GeometryShapeFunctionContainerType& ThisGeometryShapeFunctionContainer,
GeometryType* pGeometryParent)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
ThisGeometryShapeFunctionContainer)
, mpGeometryParent(pGeometryParent)
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const IntegrationPointType& ThisIntegrationPoint,
const Matrix& ThisShapeFunctionsValues,
const DenseVector<Matrix>& ThisShapeFunctionsDerivatives)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
GeometryShapeFunctionContainerType(
GeometryData::IntegrationMethod::GI_GAUSS_1,
ThisIntegrationPoint,
ThisShapeFunctionsValues,
ThisShapeFunctionsDerivatives))
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints,
const IntegrationPointType& ThisIntegrationPoint,
const Matrix& ThisShapeFunctionsValues,
const DenseVector<Matrix>& ThisShapeFunctionsDerivatives,
GeometryType* pGeometryParent)
: BaseType(ThisPoints, &mGeometryData)
, mGeometryData(
&msGeometryDimension,
GeometryShapeFunctionContainerType(
GeometryData::IntegrationMethod::GI_GAUSS_1,
ThisIntegrationPoint,
ThisShapeFunctionsValues,
ThisShapeFunctionsDerivatives))
, mpGeometryParent(pGeometryParent)
{
}

QuadraturePointGeometry(
const PointsArrayType& ThisPoints) = delete;

QuadraturePointGeometry(
const IndexType GeometryId,
const PointsArrayType& ThisPoints
) : BaseType( GeometryId, ThisPoints, &mGeometryData )
, mGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
{}, {}, {})
{
}

QuadraturePointGeometry(
const std::string& GeometryName,
const PointsArrayType& ThisPoints
) : BaseType( GeometryName, ThisPoints, &mGeometryData )
, mGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
{}, {}, {})
{
}

~QuadraturePointGeometry() override = default;

QuadraturePointGeometry(
QuadraturePointGeometry const& rOther )
: BaseType( rOther )
, mGeometryData(rOther.mGeometryData)
, mpGeometryParent(rOther.mpGeometryParent)
{
}


QuadraturePointGeometry& operator=(
const QuadraturePointGeometry& rOther )
{
BaseType::operator=( rOther );

mGeometryData = rOther.mGeometryData;
mpGeometryParent = rOther.mpGeometryParent;

return *this;
}



typename BaseType::Pointer Create( PointsArrayType const& ThisPoints ) const override
{
KRATOS_ERROR << "QuadraturePointGeometry cannot be created with 'PointsArrayType const& PointsArrayType'. "
<< "This constructor is not allowed as it would remove the evaluated shape functions as the ShapeFunctionContainer is not being copied."
<< std::endl;
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const override
{
return typename BaseType::Pointer( new QuadraturePointGeometry( NewGeometryId, rThisPoints ) );
}


typename BaseType::Pointer Create(
const IndexType NewGeometryId,
const BaseType& rGeometry
) const override
{
auto p_geometry = typename BaseType::Pointer( new QuadraturePointGeometry( NewGeometryId, rGeometry.Points() ) );
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


void Calculate(
const Variable<array_1d<double, 3>>& rVariable,
array_1d<double, 3>& rOutput) const override
{
if (rVariable == CHARACTERISTIC_GEOMETRY_LENGTH)
{
rOutput = this->IntegrationPoints()[0];
mpGeometryParent->Calculate(rVariable, rOutput);
}
}



void SetGeometryShapeFunctionContainer(
const GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>& rGeometryShapeFunctionContainer) override
{
mGeometryData.SetGeometryShapeFunctionContainer(rGeometryShapeFunctionContainer);
}


GeometryType& GetGeometryParent(IndexType Index) const override
{
return *mpGeometryParent;
}

void SetGeometryParent(GeometryType* pGeometryParent) override
{
mpGeometryParent = pGeometryParent;
}


void Calculate(
const Variable<Vector>& rVariable,
Vector& rOutput) const override
{
if (rVariable == DETERMINANTS_OF_JACOBIAN_PARENT) {
DeterminantOfJacobianParent(rOutput);
}
}


double DomainSize() const override
{
return IntegrationUtilities::ComputeDomainSize(*this);
}


Point Center() const override
{
const std::size_t node_number = this->PointsNumber();

Point point(0.0, 0.0, 0.0);
const Matrix& r_N = this->ShapeFunctionsValues();

for (IndexType point_number = 0; point_number < this->IntegrationPointsNumber(); ++point_number) {
for (IndexType i = 0; i < node_number; ++i) {
point += (*this)[i] * r_N(point_number, i);
}
}
return point;
}


SizeType PolynomialDegree(IndexType LocalDirectionIndex) const override
{
KRATOS_DEBUG_ERROR_IF_NOT(mpGeometryParent)
<< "Trying to call PolynomialDegree(LocalDirectionIndex) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->PolynomialDegree(LocalDirectionIndex);
}



CoordinatesArrayType& GlobalCoordinates(
CoordinatesArrayType& rResult,
CoordinatesArrayType const& LocalCoordinates
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call GlobalCoordinates(LocalCoordinates) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->GlobalCoordinates(rResult, LocalCoordinates);
}


CoordinatesArrayType& GlobalCoordinates(
CoordinatesArrayType& rResult,
CoordinatesArrayType const& LocalCoordinates,
Matrix& DeltaPosition
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call GlobalCoordinates(LocalCoordinates, DeltaPosition) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->GlobalCoordinates(rResult, LocalCoordinates, DeltaPosition);
}



Matrix& Jacobian(
Matrix& rResult,
const CoordinatesArrayType& rCoordinates
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call Jacobian(LocalCoordinates) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->Jacobian(rResult, rCoordinates);
}


double DeterminantOfJacobian(
const CoordinatesArrayType& rPoint
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call DeterminantOfJacobian(rPoint) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->DeterminantOfJacobian(rPoint);
}



Vector& DeterminantOfJacobianParent(
Vector& rResult) const
{
if (rResult.size() != 1)
rResult.resize(1, false);

rResult[0] = this->GetGeometryParent(0).DeterminantOfJacobian(this->IntegrationPoints()[0]);

return rResult;
}

Matrix& InverseOfJacobian(
Matrix& rResult,
const CoordinatesArrayType& rCoordinates
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call InverseOfJacobian(rPoint) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->InverseOfJacobian(rResult, rCoordinates);
}


Vector& ShapeFunctionsValues(
Vector &rResult,
const CoordinatesArrayType& rCoordinates
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call ShapeFunctionsValues(rCoordinates) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->ShapeFunctionsValues(rResult, rCoordinates);
}

virtual Matrix& ShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rPoint
) const override
{
KRATOS_DEBUG_ERROR_IF(mpGeometryParent == nullptr)
<< "Trying to call ShapeFunctionsLocalGradients(rPoint) from quadrature point. "
<< "Pointer to parent is not assigned." << std::endl;

return mpGeometryParent->ShapeFunctionsLocalGradients(rResult, rPoint);
}


GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Quadrature_Geometry;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Quadrature_Point_Geometry;
}


std::string Info() const override
{
return "Quadrature point templated by local space dimension and working space dimension.";
}

void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "Quadrature point templated by local space dimension and working space dimension.";
}

void PrintData( std::ostream& rOStream ) const override
{
}

protected:


QuadraturePointGeometry()
: BaseType(
PointsArrayType(),
&mGeometryData)
, mGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
{}, {}, {})
{
}


private:

static const GeometryDimension msGeometryDimension;


GeometryData mGeometryData;

GeometryType* mpGeometryParent = nullptr;


friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );

rSerializer.save("IntegrationPoints", mGeometryData.IntegrationPoints());
rSerializer.save("ShapeFunctionsValues", mGeometryData.ShapeFunctionsValues());
rSerializer.save("ShapeFunctionsLocalGradients", mGeometryData.ShapeFunctionsLocalGradients());
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );

IntegrationPointsContainerType integration_points;
ShapeFunctionsValuesContainerType shape_functions_values;
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients;

rSerializer.load("IntegrationPoints", integration_points[static_cast<int>(GeometryData::IntegrationMethod::GI_GAUSS_1)]);
rSerializer.load("ShapeFunctionsValues", shape_functions_values[static_cast<int>(GeometryData::IntegrationMethod::GI_GAUSS_1)]);
rSerializer.load("ShapeFunctionsLocalGradients", shape_functions_local_gradients[static_cast<int>(GeometryData::IntegrationMethod::GI_GAUSS_1)]);

mGeometryData.SetGeometryShapeFunctionContainer(GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>(
GeometryData::IntegrationMethod::GI_GAUSS_1,
integration_points,
shape_functions_values,
shape_functions_local_gradients));


}

}; 


template<class TPointType,
int TWorkingSpaceDimension,
int TLocalSpaceDimension,
int TDimension>
inline std::istream& operator >> (
std::istream& rIStream,
QuadraturePointGeometry<TPointType, TWorkingSpaceDimension, TLocalSpaceDimension, TDimension>& rThis );

template<class TPointType,
int TWorkingSpaceDimension,
int TLocalSpaceDimension,
int TDimension>
inline std::ostream& operator << (
std::ostream& rOStream,
const QuadraturePointGeometry<TPointType, TWorkingSpaceDimension, TLocalSpaceDimension, TDimension>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}


template<class TPointType,
int TWorkingSpaceDimension,
int TLocalSpaceDimension,
int TDimension>
const GeometryDimension QuadraturePointGeometry<
TPointType,
TWorkingSpaceDimension,
TLocalSpaceDimension,
TDimension>::msGeometryDimension(
TWorkingSpaceDimension,
TLocalSpaceDimension);


}  
