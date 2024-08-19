
#pragma once



#include "includes/constitutive_law.h"
#include "hyper_elastic_isotropic_neo_hookean_3d.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticIsotropicNeoHookeanPlaneStrain2D
: public HyperElasticIsotropicNeoHookean3D
{
public:


typedef ProcessInfo               ProcessInfoType;

typedef ConstitutiveLaw                CLBaseType;

typedef HyperElasticIsotropicNeoHookean3D BaseType;

typedef std::size_t                      SizeType;

typedef std::size_t                      IndexType;

static constexpr SizeType Dimension = 2;

static constexpr SizeType VoigtSize = 3;

KRATOS_CLASS_POINTER_DEFINITION( HyperElasticIsotropicNeoHookeanPlaneStrain2D );



HyperElasticIsotropicNeoHookeanPlaneStrain2D();


HyperElasticIsotropicNeoHookeanPlaneStrain2D (const HyperElasticIsotropicNeoHookeanPlaneStrain2D& rOther);


~HyperElasticIsotropicNeoHookeanPlaneStrain2D() override;




ConstitutiveLaw::Pointer Clone() const override;


void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};

protected:






void CalculateConstitutiveMatrixPK2(
Matrix& rConstitutiveMatrix,
const Matrix& InverseCTensor,
const double DeterminantF,
const double LameLambda,
const double LameMu
) override;


void CalculateConstitutiveMatrixKirchhoff(
Matrix& rConstitutiveMatrix,
const double DeterminantF,
const double LameLambda,
const double LameMu
) override;


void CalculateGreenLagrangianStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
) override;


void CalculateAlmansiStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
) override;


private:






friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType)
}


}; 
}  
