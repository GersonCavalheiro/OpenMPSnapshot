
#pragma once



#include "includes/define.h"
#include "custom_elements/base_solid_element.h"
#include "includes/variables.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacementBbar
: public BaseSolidElement
{
protected:

struct KinematicVariablesBbar
: public KinematicVariables
{
Vector Bh;


KinematicVariablesBbar(
const SizeType StrainSize,
const SizeType Dimension,
const SizeType NumberOfNodes
) : KinematicVariables(StrainSize, Dimension, NumberOfNodes)
{
Bh = ZeroVector(Dimension * NumberOfNodes);
}
};

public:
typedef ConstitutiveLaw ConstitutiveLawType;
typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;
typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef BaseSolidElement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(SmallDisplacementBbar);


SmallDisplacementBbar(IndexType NewId, GeometryType::Pointer pGeometry);
SmallDisplacementBbar(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

SmallDisplacementBbar(SmallDisplacementBbar const& rOther)
:BaseType(rOther)
{};


~SmallDisplacementBbar() override;



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


void CalculateOnIntegrationPoints(
const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

protected:



SmallDisplacementBbar() : BaseSolidElement()
{
}


bool UseElementProvidedStrain() const override;


void CalculateKinematicVariables(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod
) override;


void CalculateAndAddResidualVector(
VectorType& rRightHandSideVector,
const KinematicVariables& rThisKinematicVariables,
const ProcessInfo& rCurrentProcessInfo,
const array_1d<double, 3>& rBodyForce,
const Vector& rStressVector,
const double IntegrationWeight
) const override;


void SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
) override;



void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;


void CalculateB(
Matrix& rB,
const Matrix& DN_DX
);

Matrix ComputeEquivalentF(const Vector& rStrainTensor);


void CalculateKinematicVariablesBbar(
KinematicVariablesBbar& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
);


void CalculateBbar(
Matrix &rB,
Vector &rBh,
const Matrix &DN_DX,
const GeometryType::IntegrationPointsArrayType &IntegrationPoints,
const IndexType PointNumber
);


void CalculateHydrostaticDeformationMatrix(KinematicVariablesBbar& rThisKinematicVariables);

private:

friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
