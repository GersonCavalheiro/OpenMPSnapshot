
#pragma once



#include "includes/element.h"
#include "includes/define.h"
#include "includes/variables.h"
#include "includes/serializer.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CrBeamElement3D2N : public Element
{
protected:
static constexpr int msNumberOfNodes = 2;
static constexpr int msDimension = 3;
static constexpr unsigned int msLocalSize = msNumberOfNodes * msDimension;
static constexpr unsigned int msElementSize = msLocalSize * 2;

public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(CrBeamElement3D2N);


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

CrBeamElement3D2N() {};
CrBeamElement3D2N(IndexType NewId, GeometryType::Pointer pGeometry);
CrBeamElement3D2N(IndexType NewId, GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~CrBeamElement3D2N() override;


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


BoundedMatrix<double,msElementSize,msElementSize> CreateElementStiffnessMatrix_Material() const;


BoundedMatrix<double,msElementSize,msElementSize>  CreateElementStiffnessMatrix_Geometry() const;


virtual BoundedMatrix<double,msLocalSize,msLocalSize> CalculateDeformationStiffness() const;


BoundedMatrix<double,msElementSize,msLocalSize> CalculateTransformationS() const;


BoundedVector<double,msLocalSize> GetCurrentNodalPosition() const;


BoundedVector<double,msLocalSize> CalculateElementForces() const;


BoundedMatrix<double,msElementSize,msElementSize> CalculateInitialLocalCS() const;



BoundedMatrix<double,msDimension,msDimension> UpdateRotationMatrixLocal(Vector& Bisectrix, Vector& VectorDifference) const;

void SaveQuaternionParameters();

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void ConstCalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) const;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

virtual void ConstCalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) const;

void ConstCalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) const;

void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;



void CalculateLumpedMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) const;



void CalculateConsistentMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) const;



void BuildSingleMassMatrix(
MatrixType& rMassMatrix,
const double Phi, const double CT, const double CR, const double L, const double dir) const;

void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;

void GetValuesVector(
Vector& rValues,
int Step = 0) const override;

void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0) const override;

void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0) const override;


void AssembleSmallInBigMatrix(
const Matrix& rSmallMatrix,
BoundedMatrix<double,msElementSize,msElementSize>& rBigMatrix
) const;

int Check(const ProcessInfo& rCurrentProcessInfo) const override;



double CalculatePsi(const double I, const double A_eff) const;


double CalculateShearModulus() const;


BoundedVector<double,msElementSize> CalculateBodyForces() const;

void Calculate(const Variable<Matrix>& rVariable, Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

IntegrationMethod GetIntegrationMethod() const override;



void CalculateAndAddWorkEquivalentNodalForcesLineLoad(
const BoundedVector<double,msDimension> ForceInput,
BoundedVector<double,msElementSize>& rRightHandSideVector,
const double GeometryLength) const;



Vector CalculateSymmetricDeformationMode() const;


Vector CalculateAntiSymmetricDeformationMode() const;


Vector CalculateLocalNodalForces() const;

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

Vector CalculateGlobalNodalForces() const;

Vector GetIncrementDeformation() const;

BoundedMatrix<double, msElementSize, msElementSize> GetTransformationMatrixGlobal() const;

void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void UpdateQuaternionParameters(double& rScalNodeA,double& rScalNodeB,
Vector& rVecNodeA,Vector& rVecNodeB) const;

const Parameters GetSpecifications() const override;

private:


Vector mDeformationCurrentIteration = ZeroVector(msElementSize);
Vector mDeformationPreviousIteration = ZeroVector(msElementSize);
Vector mQuaternionVEC_A = ZeroVector(msDimension);
Vector mQuaternionVEC_B = ZeroVector(msDimension);
double mQuaternionSCA_A = 1.00;
double mQuaternionSCA_B = 1.00;




friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;

};


}
