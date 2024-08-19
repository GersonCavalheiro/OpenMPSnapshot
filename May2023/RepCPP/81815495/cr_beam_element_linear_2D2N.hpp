
#pragma once




#include "custom_elements/cr_beam_element_2D2N.hpp"
#include "includes/define.h"
#include "includes/variables.h"
#include "includes/serializer.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CrBeamElementLinear2D2N : public CrBeamElement2D2N
{
public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(CrBeamElementLinear2D2N);


CrBeamElementLinear2D2N(IndexType NewId, GeometryType::Pointer pGeometry);
CrBeamElementLinear2D2N(IndexType NewId, GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


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


~CrBeamElementLinear2D2N() override;

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;




BoundedMatrix<double,msElementSize,msElementSize> CreateRotationMatrix() override;


double CalculateLength() const override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

protected:

CrBeamElementLinear2D2N() {};

Matrix mK_Master = ZeroMatrix(msElementSize,msElementSize);

private:

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;
};

}
