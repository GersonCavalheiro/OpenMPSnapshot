
#pragma once



#include "hyper_elastic_isotropic_neo_hookean_3d.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D
: public HyperElasticIsotropicNeoHookean3D
{
public:


typedef ProcessInfo ProcessInfoType;

typedef HyperElasticIsotropicNeoHookean3D BaseType;

typedef std::size_t        SizeType;

typedef std::size_t       IndexType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

KRATOS_CLASS_POINTER_DEFINITION(HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D);



HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D();


HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D(const HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D &rOther);


~HyperElasticIsotropicQuasiIncompressibleIshochoricNeoHookean3D() override;




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


StrainMeasure GetStrainMeasure() override
{
return StrainMeasure_GreenLagrange;
}


StressMeasure GetStressMeasure() override
{
return StressMeasure_PK2;
}


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return false;
}


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;

protected:






private:





void CalculateStressAndConstitutiveMatrixPK2(
const Matrix& rC,
const double Pressure,
const double C1,
Vector& rStress,
Matrix &rTangentTensor,
const Flags& rFlags);



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
