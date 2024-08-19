
#pragma once



#include "custom_constitutive/linear_plane_stress.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) ElasticIsotropicPlaneStressUncoupledShear : public LinearPlaneStress
{
public:


typedef ProcessInfo      ProcessInfoType;
typedef LinearPlaneStress       BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION( ElasticIsotropicPlaneStressUncoupledShear );



ElasticIsotropicPlaneStressUncoupledShear();

ConstitutiveLaw::Pointer Clone() const override;


ElasticIsotropicPlaneStressUncoupledShear(const ElasticIsotropicPlaneStressUncoupledShear& rOther);


~ElasticIsotropicPlaneStressUncoupledShear() override;




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