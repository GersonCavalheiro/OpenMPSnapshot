
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{






class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ElasticIsotropic3D
: public ConstitutiveLaw
{
public:


typedef ProcessInfo      ProcessInfoType;

typedef ConstitutiveLaw         BaseType;

typedef std::size_t             SizeType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

using BaseType::Has;
using BaseType::GetValue;

KRATOS_CLASS_POINTER_DEFINITION( ElasticIsotropic3D );



ElasticIsotropic3D();


ConstitutiveLaw::Pointer Clone() const override;


ElasticIsotropic3D (const ElasticIsotropic3D& rOther);


~ElasticIsotropic3D() override;




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
return StrainMeasure_Infinitesimal;
}


StressMeasure GetStressMeasure() override
{
return StressMeasure_Cauchy;
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

protected:






void CheckClearElasticMatrix(VoigtSizeMatrixType& rConstitutiveMatrix);


virtual void CalculateElasticMatrix(
ConstitutiveLaw::VoigtSizeMatrixType& rConstitutiveMatrix,
ConstitutiveLaw::Parameters& rValues
);


virtual void CalculatePK2Stress(
const ConstitutiveLaw::StrainVectorType& rStrainVector,
ConstitutiveLaw::StressVectorType& rStressVector,
ConstitutiveLaw::Parameters& rValues
);


virtual void CalculateCauchyGreenStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw::StrainVectorType& rStrainVector
);


private:






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
