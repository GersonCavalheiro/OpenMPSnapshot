

#pragma once

#include "includes/checks.h"

#include "custom_elements/small_displacement.h"
#include "structural_mechanics_application_variables.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ZStrainDriven2p5DSmallDisplacement
: public SmallDisplacement
{
public:
typedef ConstitutiveLaw ConstitutiveLawType;
typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;
typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef SmallDisplacement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ZStrainDriven2p5DSmallDisplacement);


ZStrainDriven2p5DSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry);
ZStrainDriven2p5DSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

ZStrainDriven2p5DSmallDisplacement(ZStrainDriven2p5DSmallDisplacement const& rOther)
:BaseType(rOther)
{};

~ZStrainDriven2p5DSmallDisplacement() override = default;



void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


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


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


void SetValuesOnIntegrationPoints(
const Variable<double>& rVariable,
const std::vector<double>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;



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

std::vector<double> mImposedZStrainVector; 


ZStrainDriven2p5DSmallDisplacement() : SmallDisplacement()
{
}


void SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
) override;


private:







friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
