
#pragma once



#include "includes/define.h"
#include "includes/element.h"
#include "utilities/adjoint_extensions.h"
#include "custom_elements/total_lagrangian.h"

namespace Kratos
{


template <class TPrimalElement>
class AdjointSolidElement
: public Element
{
class ThisExtensions : public AdjointExtensions
{
Element* mpElement;

public:
explicit ThisExtensions(Element* pElement);

void GetFirstDerivativesVector(std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetSecondDerivativesVector(std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetAuxiliaryVector(std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetFirstDerivativesVariables(std::vector<VariableData const*>& rVariables) const override;

void GetSecondDerivativesVariables(std::vector<VariableData const*>& rVariables) const override;

void GetAuxiliaryVariables(std::vector<VariableData const*>& rVariables) const override;
};

public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AdjointSolidElement);


AdjointSolidElement(IndexType NewId = 0);

AdjointSolidElement(IndexType NewId, GeometryType::Pointer pGeometry);

AdjointSolidElement(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


Element::Pointer Create(IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override;

Element::Pointer Create(IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateFirstDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateSecondDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void GetValuesVector(Vector& rValues, int Step = 0) const override;

void EquationIdVector(EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo) const override;

void GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const override;

int Check(const ProcessInfo& rCurrentProcessInfo) const override;

void CalculateSensitivityMatrix(const Variable<array_1d<double, 3>>& rDesignVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


protected:

private:

TPrimalElement mPrimalElement;


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
rSerializer.save("mPrimalElement", mPrimalElement);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
rSerializer.load("mPrimalElement", mPrimalElement);
}
};


} 
