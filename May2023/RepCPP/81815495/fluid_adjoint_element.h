
#pragma once



#include "includes/checks.h"
#include "includes/constitutive_law.h"
#include "includes/element.h"
#include "includes/properties.h"
#include "utilities/adjoint_extensions.h"


namespace Kratos
{

template <unsigned int TDim, unsigned int TNumNodes, class TAdjointElementData>
class FluidAdjointElement : public Element
{
class ThisExtensions : public AdjointExtensions
{
Element* mpElement;

public:
explicit ThisExtensions(Element* pElement);

void GetFirstDerivativesVector(
std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetSecondDerivativesVector(
std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetAuxiliaryVector(
std::size_t NodeId,
std::vector<IndirectScalar<double>>& rVector,
std::size_t Step) override;

void GetFirstDerivativesVariables(std::vector<VariableData const*>& rVariables) const override;

void GetSecondDerivativesVariables(std::vector<VariableData const*>& rVariables) const override;

void GetAuxiliaryVariables(std::vector<VariableData const*>& rVariables) const override;
};

public:

using BaseType = Element;

using NodesArrayType = typename BaseType::NodesArrayType;

using PropertiesType = typename BaseType::PropertiesType;

using GeometryType = typename BaseType::GeometryType;

using VectorType = typename BaseType::VectorType;

using MatrixType = typename BaseType::MatrixType;

using IndexType = std::size_t;

using ShapeFunctionDerivativesArrayType = GeometryType::ShapeFunctionsGradientsType;

constexpr static IndexType TBlockSize = TDim + 1;

constexpr static IndexType TElementLocalSize = TBlockSize * TNumNodes;

constexpr static IndexType TCoordsLocalSize = TDim * TNumNodes;

using VectorF = BoundedVector<double, TElementLocalSize>;

KRATOS_CLASS_POINTER_DEFINITION(FluidAdjointElement);



FluidAdjointElement(IndexType NewId = 0);


FluidAdjointElement(
IndexType NewId,
GeometryType::Pointer pGeometry);


FluidAdjointElement(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~FluidAdjointElement() override;





Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties) const override;


Element::Pointer Clone(
IndexType NewId,
NodesArrayType const& ThisNodes) const override;

int Check(const ProcessInfo& rCurrentProcessInfo) const override;


void EquationIdVector(
EquationIdVectorType& rElementalEquationIdList,
const ProcessInfo& rCurrentProcessInfo) const override;


void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo) const override;

void GetValuesVector(
VectorType& rValues,
int Step = 0) const override;

void GetFirstDerivativesVector(
VectorType& rValues,
int Step = 0) const override;

void GetSecondDerivativesVector(
VectorType& rValues,
int Step) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLocalVelocityContribution(
MatrixType &rDampMatrix,
VectorType &rRightHandSideVector,
const ProcessInfo &rCurrentProcessInfo) override;

void CalculateFirstDerivativesLHS(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateSecondDerivativesLHS(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateSensitivityMatrix(
const Variable<array_1d<double, 3>>& rSensitivityVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(
const Variable<Vector>& rVariable,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;


protected:

ConstitutiveLaw::Pointer mpConstitutiveLaw = nullptr;



void AddFluidResidualsContributions(
VectorType& rResidual,
const ProcessInfo& rCurrentProcessInfo);


void AddFluidFirstDerivatives(
MatrixType& rDerivativesMatrix,
const ProcessInfo& rCurrentProcessInfo,
const double MassTermsDerivativesWeight = 1.0);


void AddFluidSecondDerivatives(
MatrixType& rDerivativesMatrix,
const ProcessInfo& rCurrentProcessInfo);


void AddFluidShapeDerivatives(
MatrixType& rDerivativesMatrix,
const ProcessInfo& rCurrentProcessInfo);


void CalculateGeometryData(
Vector& rGaussWeights,
Matrix& rNContainer,
ShapeFunctionDerivativesArrayType& rDN_DX,
const GeometryData::IntegrationMethod& rIntegrationMethod) const;

};


} 