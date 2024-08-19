
#pragma once



#include "includes/constitutive_law.h"
#include "includes/checks.h"

namespace Kratos
{




class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) TrussPlasticityConstitutiveLaw : public ConstitutiveLaw
{
public:

typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION( TrussPlasticityConstitutiveLaw );




TrussPlasticityConstitutiveLaw();

ConstitutiveLaw::Pointer Clone() const override;


TrussPlasticityConstitutiveLaw (const TrussPlasticityConstitutiveLaw& rOther);



~TrussPlasticityConstitutiveLaw() override;






void GetLawFeatures(Features& rFeatures) override;

void SetValue(const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo) override;

double& GetValue(const Variable<double>& rThisVariable,
double& rValue) override;

array_1d<double, 3 > & GetValue(const Variable<array_1d<double, 3 > >& rThisVariable,
array_1d<double, 3 > & rValue) override;

double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


double TrialYieldFunction(const Properties& rMaterialProperties,
const double& rCurrentStress);


bool CheckIfIsPlasticRegime(Parameters& rValues,const double& rCurrentStress);

void FinalizeMaterialResponsePK2(Parameters& rValues) override;

void CalculateMaterialResponsePK2(Parameters& rValues) override;

void CalculateMaterialResponsePK2Custom(Parameters& rValues, double& rCurrentAccumulatedPlasticStrain, double& rCurrentPlasticAlpha);

bool RequiresInitializeMaterialResponse() override
{
return false;
}


SizeType GetStrainSize() const override
{
return 1;
}


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;

protected:





private:


bool mCurrentInElasticFlag = false;
double mPlasticAlpha = 0.0;
double mAccumulatedPlasticStrain = 0.0;





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw);
rSerializer.save("PlasticAlpha", mPlasticAlpha);
rSerializer.save("AccumulatedPlasticStrain", mAccumulatedPlasticStrain);
rSerializer.save("CurrentInElasticFlag", mCurrentInElasticFlag);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw);
rSerializer.load("PlasticAlpha", mPlasticAlpha);
rSerializer.load("AccumulatedPlasticStrain", mAccumulatedPlasticStrain);
rSerializer.load("CurrentInElasticFlag", mCurrentInElasticFlag);
}


}; 
}  
