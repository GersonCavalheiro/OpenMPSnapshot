
#pragma once



#include "includes/constitutive_law.h"

#include "hyper_elastic_isotropic_neo_hookean_3d.h"

namespace Kratos
{







class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticSimoTaylorNeoHookean3D
: public HyperElasticIsotropicNeoHookean3D
{
public:


typedef ProcessInfo ProcessInfoType;

typedef HyperElasticIsotropicNeoHookean3D    BaseType;

typedef std::size_t        SizeType;

typedef std::size_t       IndexType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

KRATOS_CLASS_POINTER_DEFINITION(HyperElasticSimoTaylorNeoHookean3D);



HyperElasticSimoTaylorNeoHookean3D();


HyperElasticSimoTaylorNeoHookean3D (const HyperElasticSimoTaylorNeoHookean3D& rOther);


~HyperElasticSimoTaylorNeoHookean3D() override;




ConstitutiveLaw::Pointer Clone() const override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};


StrainMeasure GetStrainMeasure() override
{
return StrainMeasure_GreenLagrange;
}


StressMeasure GetStressMeasure() override
{
return StressMeasure_PK2;
}


void CalculateMaterialResponsePK2 (ConstitutiveLaw::Parameters & rValues) override;


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;

protected:






private:






virtual void AuxiliaryCalculateConstitutiveMatrixPK2(
Matrix& rConstitutiveMatrix,
const Vector& rStrain,
const double Kappa,
const double Mu);


virtual void AuxiliaryCalculatePK2Stress(
Vector& rStressVector,
const Vector& rStrain,
const double Kappa,
const double Mu);


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw)
}


}; 
}  