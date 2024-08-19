
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AxisymElasticIsotropic : public ElasticIsotropic3D
{
public:


typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION( AxisymElasticIsotropic );



AxisymElasticIsotropic();

ConstitutiveLaw::Pointer Clone() const override;


AxisymElasticIsotropic (const AxisymElasticIsotropic& rOther);


~AxisymElasticIsotropic() override;




void GetLawFeatures(Features& rFeatures) override;


SizeType GetStrainSize() const override
{
return 4;
}






protected:






private:





void CalculateElasticMatrix(VoigtSizeMatrixType& C, ConstitutiveLaw::Parameters& rValues) override;


void CalculatePK2Stress(const Vector &rStrainVector, ConstitutiveLaw::StressVectorType &rStressVector, ConstitutiveLaw::Parameters &rValues) override;


void CalculateCauchyGreenStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw::StrainVectorType& rStrainVector
) override;




friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw)
}


}; 
}  
