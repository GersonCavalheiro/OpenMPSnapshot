
#pragma once



#include "includes/constitutive_law.h"

#include "hyper_elastic_simo_taylor_neo_hookean_3d.h"

namespace Kratos
{







class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticSimoTaylorNeoHookeanPlaneStrain2D
: public HyperElasticSimoTaylorNeoHookean3D
{
public:


typedef HyperElasticSimoTaylorNeoHookean3D BaseType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

static constexpr SizeType Dimension = 2;

static constexpr SizeType VoigtSize = 3;

KRATOS_CLASS_POINTER_DEFINITION(HyperElasticSimoTaylorNeoHookeanPlaneStrain2D);



HyperElasticSimoTaylorNeoHookeanPlaneStrain2D();


HyperElasticSimoTaylorNeoHookeanPlaneStrain2D (const HyperElasticSimoTaylorNeoHookeanPlaneStrain2D& rOther);


~HyperElasticSimoTaylorNeoHookeanPlaneStrain2D() override;




ConstitutiveLaw::Pointer Clone() const override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};

protected:






void CalculateGreenLagrangianStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector) override;

private:





void AuxiliaryCalculateConstitutiveMatrixPK2(
Matrix& rConstitutiveMatrix,
const Vector& rStrain,
const double Kappa,
const double Mu) override;


void AuxiliaryCalculatePK2Stress(
Vector& rStressVector,
const Vector& rStrain,
const double Kappa,
const double Mu) override;




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
