
#pragma once



#include "includes/element.h"
#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TrussElement3D2N : public Element
{
protected:
static constexpr int msNumberOfNodes = 2;
static constexpr int msDimension = 3;
static constexpr unsigned int msLocalSize = msNumberOfNodes * msDimension;
ConstitutiveLaw::Pointer mpConstitutiveLaw = nullptr;

public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TrussElement3D2N);


typedef Element BaseType;
typedef BaseType::GeometryType GeometryType;
typedef BaseType::NodesArrayType NodesArrayType;
typedef BaseType::PropertiesType PropertiesType;
typedef BaseType::IndexType IndexType;
typedef BaseType::SizeType SizeType;
typedef BaseType::MatrixType MatrixType;
typedef BaseType::VectorType VectorType;
typedef BaseType::EquationIdVectorType EquationIdVectorType;
typedef BaseType::DofsVectorType DofsVectorType;


TrussElement3D2N() {};
TrussElement3D2N(IndexType NewId,
GeometryType::Pointer pGeometry);
TrussElement3D2N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~TrussElement3D2N() override;


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

void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo) const override;

void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


virtual BoundedMatrix<double,msLocalSize,msLocalSize>
CreateElementStiffnessMatrix(const ProcessInfo& rCurrentProcessInfo);

void Calculate(const Variable<Matrix>& rVariable, Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(const Variable<double>& rVariable, double& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


virtual void UpdateInternalForces(BoundedVector<double,msLocalSize>& rInternalForces, const ProcessInfo& rCurrentProcessInfo);


void CreateTransformationMatrix(BoundedMatrix<double,msLocalSize,msLocalSize>& rRotationMatrix);


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

void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateConsistentMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) const;

void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;



void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void GetValuesVector(
Vector& rValues,
int Step = 0) const override;

void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0) const override;

void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0) const override;

int  Check(
const ProcessInfo& rCurrentProcessInfo) const override;


double CalculateGreenLagrangeStrain()const;


BoundedVector<double,msLocalSize> CalculateBodyForces();


void CalculateGeometricStiffnessMatrix(BoundedMatrix<double,msLocalSize,msLocalSize>& rGeometricStiffnessMatrix,
const ProcessInfo& rCurrentProcessInfo);


void CalculateElasticStiffnessMatrix(BoundedMatrix<double,msLocalSize,msLocalSize>& rElasticStiffnessMatrix,
const ProcessInfo& rCurrentProcessInfo);


virtual void WriteTransformationCoordinates(
BoundedVector<double,msLocalSize>& rReferenceCoordinates);

double ReturnTangentModulus1D(const ProcessInfo& rCurrentProcessInfo);

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


bool HasSelfWeight() const;

const Parameters GetSpecifications() const override;

private:

void CalculateLumpedMassVector(
VectorType& rMassVector,
const ProcessInfo& rCurrentProcessInfo) const override;

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;
};


}
