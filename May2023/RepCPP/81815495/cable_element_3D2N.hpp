
#pragma once



#include "custom_elements/truss_element_3D2N.hpp"
#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos
{

class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CableElement3D2N : public TrussElement3D2N
{

public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(CableElement3D2N);



CableElement3D2N() {};
CableElement3D2N(IndexType NewId,
GeometryType::Pointer pGeometry);
CableElement3D2N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~CableElement3D2N() override;


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

BoundedMatrix<double,msLocalSize,msLocalSize>
CreateElementStiffnessMatrix(const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void UpdateInternalForces(BoundedVector<double,msLocalSize>& rinternalForces, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector<array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable, std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

private:
bool mIsCompressed;

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;
};

}
