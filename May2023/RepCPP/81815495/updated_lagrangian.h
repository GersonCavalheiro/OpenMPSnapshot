
#pragma once



#include "includes/define.h"
#include "custom_elements/base_solid_element.h"
#include "includes/variables.h"
#include "includes/constitutive_law.h"

namespace Kratos
{




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) UpdatedLagrangian
: public BaseSolidElement
{
public:

typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef BaseSolidElement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(UpdatedLagrangian);


UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry);
UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

UpdatedLagrangian(UpdatedLagrangian const& rOther)
:BaseType(rOther)
,mF0Computed(rOther.mF0Computed)
,mDetF0(rOther.mDetF0)
,mF0(rOther.mF0)
{};

~UpdatedLagrangian() override;



void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


Element::Pointer Clone (
IndexType NewId,
NodesArrayType const& rThisNodes
) const override;


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
const Variable<bool>& rVariable,
std::vector<bool>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<int>& rVariable,
std::vector<int>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector<array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 6>>& rVariable,
std::vector<array_1d<double, 6>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<double>& rVariable,
const std::vector<double>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<Matrix>& rVariable,
const std::vector<Matrix>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;



std::string Info() const override
{
std::stringstream buffer;
buffer << "Updated Lagrangian Solid Element #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Updated Lagrangian Solid Element #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:



bool mF0Computed;           
std::vector<double> mDetF0; 
std::vector<Matrix> mF0;    


UpdatedLagrangian() : BaseSolidElement()
{
}


void CloneUpdatedLagrangianDatabase(
const bool rF0Computed,
const std::vector<double>& rDetF0,
const std::vector<Matrix>& rF0
)
{
mF0Computed = rF0Computed;
mDetF0 = rDetF0;
mF0 = rF0;
}


ConstitutiveLaw::StressMeasure GetStressMeasure() const override;


void UpdateHistoricalDatabase(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber
);


void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;


void CalculateKinematicVariables(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod
) override;


double CalculateDerivativesOnReferenceConfiguration(
Matrix& J0,
Matrix& InvJ0,
Matrix& DN_DX,
const IndexType PointNumber,
IntegrationMethod ThisIntegrationMethod
) const override;


private:




void CalculateB(
Matrix& rB,
const Matrix& rDN_DX,
const SizeType StrainSize,
const IndexType PointNumber
);


double ReferenceConfigurationDeformationGradientDeterminant(const IndexType PointNumber) const;


Matrix ReferenceConfigurationDeformationGradient(const IndexType PointNumber) const;




friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
