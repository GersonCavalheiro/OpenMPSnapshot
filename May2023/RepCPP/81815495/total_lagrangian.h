
#pragma once



#include "includes/define.h"
#include "custom_elements/base_solid_element.h"
#include "includes/variables.h"

namespace Kratos
{




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TotalLagrangian
: public BaseSolidElement
{
public:

typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef BaseSolidElement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TotalLagrangian);


TotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry);
TotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

TotalLagrangian(TotalLagrangian const& rOther)
:BaseType(rOther)
{};

~TotalLagrangian() override;



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


Element::Pointer Clone (
IndexType NewId,
NodesArrayType const& rThisNodes
) const override;


void CalculateSensitivityMatrix(const Variable<array_1d<double, 3>>& rDesignVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;



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


TotalLagrangian() : BaseSolidElement()
{
}


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


std::size_t GetStrainSize() const;


private:





void CalculateB(Matrix& rB, Matrix const& rF, const Matrix& rDN_DX);

void Calculate2DB(Matrix& rB, const Matrix& rF, const Matrix& rDN_DX);

void Calculate3DB(Matrix& rB, const Matrix& rF, const Matrix& rDN_DX);

void CalculateAxisymmetricB(Matrix& rB, const Matrix& rF, const Matrix& rDN_DX, const Vector& rN);

void CalculateAxisymmetricF(Matrix const& rJ, Matrix const& rInvJ0, Vector const& rN, Matrix& rF);

void CalculateStress(Vector& rStrain,
std::size_t IntegrationPoint,
Vector& rStress,
ProcessInfo const& rCurrentProcessInfo);

void CalculateStress(Matrix const& rF,
std::size_t IntegrationPoint,
Vector& rStress,
ProcessInfo const& rCurrentProcessInfo);

void CalculateStrain(Matrix const& rF,
std::size_t IntegrationPoint,
Vector& rStrain,
ProcessInfo const& rCurrentProcessInfo);

void CalculateShapeSensitivity(ShapeParameter Deriv,
Matrix& rDN_DX0,
Matrix& rDN_DX0_Deriv,
Matrix& rF_Deriv,
double& rDetJ0_Deriv,
std::size_t IntegrationPointIndex);

void CalculateBSensitivity(Matrix const& rDN_DX,
Matrix const& rF,
Matrix const& rDN_DX_Deriv,
Matrix const& rF_Deriv,
Matrix& rB_Deriv);


bool IsAxissymmetric() const;


friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
