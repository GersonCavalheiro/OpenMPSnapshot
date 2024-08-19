

#pragma once





#include "includes/define.h"
#include "custom_elements/total_lagrangian.h"
#include "includes/variables.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TotalLagrangianQ1P0MixedElement
: public TotalLagrangian
{
public:

typedef TotalLagrangian BaseType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TotalLagrangianQ1P0MixedElement);


TotalLagrangianQ1P0MixedElement(IndexType NewId, GeometryType::Pointer pGeometry);
TotalLagrangianQ1P0MixedElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

TotalLagrangianQ1P0MixedElement(TotalLagrangianQ1P0MixedElement const& rOther)
:BaseType(rOther)
{};

~TotalLagrangianQ1P0MixedElement() override;



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
buffer << "TotalLagrangianQ1P0MixedElement #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "TotalLagrangianQ1P0MixedElement #" << Id() << "\nConstitutive law: " << BaseType::mConstitutiveLawVector[0]->Info();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:


TotalLagrangianQ1P0MixedElement() : TotalLagrangian()
{
}


void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


double CalculateBulkModulus(
const Properties &rProperties);


int Check( const ProcessInfo& rCurrentProcessInfo ) const override;


void FinalizeNonLinearIteration(const ProcessInfo &rCurrentProcessInfo) override;


private:





friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
