
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) HyperElasticIsotropicKirchhoff3D
: public ConstitutiveLaw
{
public:


typedef ProcessInfo      ProcessInfoType;

typedef ConstitutiveLaw         BaseType;

typedef std::size_t             SizeType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

KRATOS_CLASS_POINTER_DEFINITION( HyperElasticIsotropicKirchhoff3D );



HyperElasticIsotropicKirchhoff3D();

ConstitutiveLaw::Pointer Clone() const override;


HyperElasticIsotropicKirchhoff3D (const HyperElasticIsotropicKirchhoff3D& rOther);


~HyperElasticIsotropicKirchhoff3D() override;




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

protected:






virtual void CalculateConstitutiveMatrixPK2(
Matrix& rConstitutiveMatrix,
const double YoungModulus,
const double PoissonCoefficient
);


virtual void CalculateConstitutiveMatrixKirchhoff(
Matrix& rConstitutiveMatrix,
const Matrix& rDeformationGradientF,
const double YoungModulus,
const double PoissonCoefficient
);


virtual void CalculatePK2Stress(
const Vector& rStrainVector,
Vector& rStressVector,
const double YoungModulus,
const double PoissonCoefficient
);


virtual void CalculateKirchhoffStress(
const Vector& rStrainVector,
Vector& rStressVector,
const Matrix& rDeformationGradientF,
const double YoungModulus,
const double PoissonCoefficient
);


virtual void CalculateGreenLagrangianStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
);


virtual void CalculateAlmansiStrain(
ConstitutiveLaw::Parameters& rValues,
Vector& rStrainVector
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
