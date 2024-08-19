
#pragma once



#include "custom_elements/updated_lagrangian.h"

namespace Kratos
{






class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AxisymUpdatedLagrangian
: public UpdatedLagrangian
{
public:
typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AxisymUpdatedLagrangian);


AxisymUpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry);
AxisymUpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

~AxisymUpdatedLagrangian() override;


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







protected:


AxisymUpdatedLagrangian() : UpdatedLagrangian()
{
}


private:





double GetIntegrationWeight(
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const IndexType PointNumber,
const double detJ
) const override;



friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
