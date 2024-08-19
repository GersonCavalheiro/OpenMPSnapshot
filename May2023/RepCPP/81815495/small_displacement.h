
#pragma once



#include "includes/define.h"
#include "custom_elements/base_solid_element.h"
#include "includes/variables.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacement
: public BaseSolidElement
{
public:
typedef ConstitutiveLaw ConstitutiveLawType;
typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;
typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef BaseSolidElement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(SmallDisplacement);


SmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry);
SmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

SmallDisplacement(SmallDisplacement const& rOther)
:BaseType(rOther)
{};

~SmallDisplacement() override;



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



std::string Info() const override
{
std::stringstream buffer;
buffer << "Small Displacement Solid Element #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Small Displacement Solid Element #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:


SmallDisplacement() : BaseSolidElement()
{
}


bool UseElementProvidedStrain() const override;


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


void SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
) override;


virtual void CalculateB(
Matrix& rB,
const Matrix& rDN_DX,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const IndexType PointNumber
) const;


virtual void ComputeEquivalentF(
Matrix& rF,
const Vector& StrainVector
) const;


private:







friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
