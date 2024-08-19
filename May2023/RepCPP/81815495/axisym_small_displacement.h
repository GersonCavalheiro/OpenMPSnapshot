
#pragma once



#include "custom_elements/small_displacement.h"

namespace Kratos
{






class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AxisymSmallDisplacement
: public SmallDisplacement
{
public:
typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AxisymSmallDisplacement);


AxisymSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry);
AxisymSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

~AxisymSmallDisplacement() override;


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


AxisymSmallDisplacement() : SmallDisplacement()
{
}


private:





void CalculateB(
Matrix& rB,
const Matrix& DN_DX,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const IndexType PointNumber
) const override;


void ComputeEquivalentF(
Matrix& rF,
const Vector& StrainVector
) const override;


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
