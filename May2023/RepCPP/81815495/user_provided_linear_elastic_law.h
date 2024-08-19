
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{







template <unsigned int TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) UserProvidedLinearElasticLaw
: public ConstitutiveLaw
{
public:


typedef ProcessInfo      ProcessInfoType;

typedef ConstitutiveLaw         BaseType;

typedef std::size_t             SizeType;

static constexpr SizeType Dimension = TDim;

static constexpr SizeType StrainSize = (TDim * 3) - 3;

KRATOS_CLASS_POINTER_DEFINITION( UserProvidedLinearElasticLaw );



UserProvidedLinearElasticLaw();


ConstitutiveLaw::Pointer Clone() const override;


UserProvidedLinearElasticLaw (const UserProvidedLinearElasticLaw& rOther);


~UserProvidedLinearElasticLaw();




void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return StrainSize;
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


void InitializeMaterialResponsePK1 (Parameters& rValues) override {};


void InitializeMaterialResponsePK2 (Parameters& rValues) override {};


void InitializeMaterialResponseKirchhoff (Parameters& rValues) override {};


void InitializeMaterialResponseCauchy (Parameters& rValues) override {};


void FinalizeMaterialResponsePK1 (Parameters& rValues) override {};


void FinalizeMaterialResponsePK2 (Parameters& rValues) override {};


void FinalizeMaterialResponseKirchhoff (Parameters& rValues) override {};



void FinalizeMaterialResponseCauchy (Parameters& rValues) override {};


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue) override;




Matrix& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue) override;




int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo) const override;

protected:






virtual void CalculateElasticMatrix(
ConstitutiveLaw::VoigtSizeMatrixType& rConstitutiveMatrix,
ConstitutiveLaw::Parameters& rValues
);


virtual void CalculatePK2Stress(
const ConstitutiveLaw::StrainVectorType& rStrainVector,
ConstitutiveLaw::StressVectorType& rStressVector,
ConstitutiveLaw::Parameters& rValues
);


virtual void CalculateGreenLagrangeStrainVector(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw::StrainVectorType& rStrainVector
);


private:






friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
}


}; 
}  
