
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) LinearPlaneStress
: public ElasticIsotropic3D
{
public:

typedef ProcessInfo      ProcessInfoType;

typedef ConstitutiveLaw       CLBaseType;

typedef ElasticIsotropic3D      BaseType;

using BaseType::Has;
using BaseType::GetValue;

typedef std::size_t             SizeType;

static constexpr SizeType Dimension = 2;

static constexpr SizeType VoigtSize = 3;

KRATOS_CLASS_POINTER_DEFINITION( LinearPlaneStress );



LinearPlaneStress();

ConstitutiveLaw::Pointer Clone() const override;


LinearPlaneStress (const LinearPlaneStress& rOther);



~LinearPlaneStress() override;




void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
}






bool& GetValue(const Variable<bool>& rThisVariable, bool& rValue) override;


protected:






void CalculateElasticMatrix(VoigtSizeMatrixType& C, ConstitutiveLaw::Parameters& rValues) override;


void CalculatePK2Stress(
const ConstitutiveLaw::StrainVectorType& rStrainVector,
ConstitutiveLaw::StressVectorType& rStressVector,
ConstitutiveLaw::Parameters& rValues
) override;


void CalculateCauchyGreenStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw::StrainVectorType& rStrainVector
) override;


private:






friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ElasticIsotropic3D)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ElasticIsotropic3D)
}


}; 
}  
