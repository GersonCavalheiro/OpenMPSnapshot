
#pragma once



#include "custom_elements/truss_element_3D2N.hpp"
#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos {


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TrussElementLinear3D2N : public TrussElement3D2N
{
public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TrussElementLinear3D2N);

TrussElementLinear3D2N() {};
TrussElementLinear3D2N(IndexType NewId,
GeometryType::Pointer pGeometry);
TrussElementLinear3D2N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~TrussElementLinear3D2N() override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


void AddPrestressLinear(VectorType& rRightHandSideVector);

void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable, std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


BoundedMatrix<double,msLocalSize,msLocalSize>
CreateElementStiffnessMatrix(const ProcessInfo& rCurrentProcessInfo) override;


void WriteTransformationCoordinates(
BoundedVector<double,msLocalSize>& rReferenceCoordinates) override;


double CalculateLinearStrain();




void UpdateInternalForces(
BoundedVector<double,msLocalSize>& rInternalForces, const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

private:

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;
};


}
