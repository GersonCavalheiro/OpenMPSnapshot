
#pragma once


#include "custom_constitutive/linear_plane_stress.h"

namespace Kratos
{


class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) MultiLinearIsotropicPlaneStress2D
: public LinearPlaneStress
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MultiLinearIsotropicPlaneStress2D );



MultiLinearIsotropicPlaneStress2D();

ConstitutiveLaw::Pointer Clone() const override;


MultiLinearIsotropicPlaneStress2D (const MultiLinearIsotropicPlaneStress2D& rOther);



~MultiLinearIsotropicPlaneStress2D() override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;



protected:


void CalculateElasticMatrix(Matrix& C, ConstitutiveLaw::Parameters& rValues) override;


void CalculatePK2Stress(
const Vector& rStrainVector,
Vector& rStressVector,
ConstitutiveLaw::Parameters& rValues
) override;


private:

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, LinearPlaneStress)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, LinearPlaneStress)
}


}; 
}  
