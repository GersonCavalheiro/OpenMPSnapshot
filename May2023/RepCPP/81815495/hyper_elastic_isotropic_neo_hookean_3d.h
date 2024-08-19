
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticIsotropicNeoHookean3D
: public ConstitutiveLaw
{
public:


typedef ProcessInfo ProcessInfoType;

typedef ConstitutiveLaw    BaseType;

typedef std::size_t        SizeType;

typedef std::size_t       IndexType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

KRATOS_CLASS_POINTER_DEFINITION( HyperElasticIsotropicNeoHookean3D );



HyperElasticIsotropicNeoHookean3D();


HyperElasticIsotropicNeoHookean3D (const HyperElasticIsotropicNeoHookean3D& rOther);


~HyperElasticIsotropicNeoHookean3D() override;




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


void CalculateMaterialResponsePK1 (ConstitutiveLaw::Parameters & rValues) override;


void CalculateMaterialResponsePK2 (ConstitutiveLaw::Parameters & rValues) override;


void CalculateMaterialResponseKirchhoff (ConstitutiveLaw::Parameters & rValues) override;


void CalculateMaterialResponseCauchy (ConstitutiveLaw::Parameters & rValues) override;


void InitializeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1 (ConstitutiveLaw::Parameters & rValues) override;


void FinalizeMaterialResponsePK2 (ConstitutiveLaw::Parameters & rValues) override;


void FinalizeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters & rValues)  override;


void FinalizeMaterialResponseCauchy (ConstitutiveLaw::Parameters & rValues) override;


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return false;
}


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue
) override;


Vector& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


Matrix& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


virtual void CalculateGreenLagrangianStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
);


virtual void CalculateAlmansiStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
);

protected:






private:





virtual void CalculateConstitutiveMatrixPK2(
Matrix& rConstitutiveMatrix,
const Matrix& rInverseCTensor,
const double DeterminantF,
const double LameLambda,
const double LameMu
);


virtual void CalculateConstitutiveMatrixKirchhoff(
Matrix& rConstitutiveMatrix,
const double DeterminantF,
const double LameLambda,
const double LameMu
);


virtual void CalculatePK2Stress(
const Matrix& rInvCTensor,
Vector& rStressVector,
const double DeterminantF,
const double LameLambda,
const double LameMu
);


virtual void CalculateKirchhoffStress(ConstitutiveLaw::Parameters& rValues);



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
