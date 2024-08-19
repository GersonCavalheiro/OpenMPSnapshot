
#pragma once




#include "includes/element.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) NodalConcentratedElement
: public Element
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( NodalConcentratedElement);

public:


NodalConcentratedElement(IndexType NewId, GeometryType::Pointer pGeometry, bool UseRayleighDamping = false);

NodalConcentratedElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties, bool UseRayleighDamping = false);

NodalConcentratedElement(NodalConcentratedElement const& rOther);

~NodalConcentratedElement() override;


NodalConcentratedElement& operator=(NodalConcentratedElement const& rOther);


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


Element::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override;

/
void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetValuesVector(Vector& rValues, int Step = 0) const override;


void GetFirstDerivativesVector(Vector& rValues, int Step = 0) const override;


void GetSecondDerivativesVector(Vector& rValues, int Step = 0) const override;


/

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;




void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;



void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;


void AddExplicitContribution(
const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3>>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;


protected:

bool mUseRayleighDamping;

NodalConcentratedElement() : Element()
{
}




Matrix& CalculateDeltaPosition(Matrix & rDeltaPosition);


private:

friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;



}; 


} 
