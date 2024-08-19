
#pragma once



#include "includes/element.h"
#include "includes/define.h"
#include "includes/variables.h"
#include "includes/serializer.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CrBeamElement2D2N : public Element
{
protected:
static constexpr int msNumberOfNodes = 2;
static constexpr int msDimension = 2;
static constexpr unsigned int msLocalSize = 3;
static constexpr unsigned int msElementSize = msLocalSize * 2;

BoundedVector<double,msLocalSize> mDeformationForces = ZeroVector(msLocalSize);


Vector mInternalGlobalForces = ZeroVector(msElementSize);

public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(CrBeamElement2D2N);


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

CrBeamElement2D2N() {};
CrBeamElement2D2N(IndexType NewId, GeometryType::Pointer pGeometry);
CrBeamElement2D2N(IndexType NewId, GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~CrBeamElement2D2N() override;



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

void GetValuesVector(
Vector& rValues,
int Step = 0) const override;

void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0) const override;

void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0) const override;

void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

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

void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;

int Check(const ProcessInfo& rCurrentProcessInfo) const override;



double CalculateShearModulus() const;



double CalculatePsi(const double I, const double A_eff) const;


double CalculateInitialElementAngle() const;


double CalculateDeformedElementAngle();


BoundedVector<double,msElementSize> CalculateBodyForces() const;


void CalculateAndAddWorkEquivalentNodalForcesLineLoad(
const BoundedVector<double,3> ForceInput,
BoundedVector<double,msElementSize>& rRightHandSideVector,
const double GeometryLength)const ;

IntegrationMethod GetIntegrationMethod() const override;


BoundedMatrix<double,msElementSize,msLocalSize> CalculateTransformationS() const;


virtual double CalculateLength() const;


BoundedMatrix<double,msLocalSize,msLocalSize> CreateElementStiffnessMatrix_Kd_mat() const;


BoundedMatrix<double,msLocalSize,msLocalSize> CreateElementStiffnessMatrix_Kd_geo() const;


BoundedMatrix<double,msElementSize,msElementSize> CreateElementStiffnessMatrix_Kr() const;


BoundedMatrix<double,msElementSize,msElementSize> CreateElementStiffnessMatrix_Total() const;



void GlobalizeMatrix(Matrix& A);


void GlobalizeVector(Vector& A);


double Modulus2Pi(double A) const;


virtual BoundedMatrix<double,msElementSize,msElementSize> CreateRotationMatrix();



BoundedVector<double,msLocalSize> CalculateDeformationParameters();


BoundedVector<double,msLocalSize> CalculateInternalStresses_DeformationModes();


BoundedVector<double,msElementSize> ReturnElementForces_Local();




void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

const Parameters GetSpecifications() const override;

private:

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;

};

}
