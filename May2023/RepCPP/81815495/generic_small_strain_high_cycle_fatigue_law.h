
#pragma once



#include "custom_constitutive/small_strains/damage/generic_small_strain_isotropic_damage.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainHighCycleFatigueLaw
: public GenericSmallStrainIsotropicDamage<TConstLawIntegratorType>
{
public:

static constexpr SizeType Dimension = TConstLawIntegratorType::Dimension;

static constexpr SizeType VoigtSize = TConstLawIntegratorType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainHighCycleFatigueLaw);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();
static constexpr double threshold_tolerance = 1.0e-5;

typedef GenericSmallStrainIsotropicDamage<TConstLawIntegratorType> BaseType;



GenericSmallStrainHighCycleFatigueLaw()
{
}

GenericSmallStrainHighCycleFatigueLaw(  const double FatigueReductionFactor,
const double PreviousStress0,
const double PreviousStress1,
const double MaxStress,
const double MinStress,
const unsigned int NumberOfCyclesGlobal,
const double FatigueReductionParameter)
{
mFatigueReductionFactor = FatigueReductionFactor;
Vector PreviousStresses = ZeroVector(2);
PreviousStresses[0] = PreviousStress0;
PreviousStresses[1] = PreviousStress1;
mPreviousStresses = PreviousStresses;
mMaxStress = MaxStress;
mMinStress = MinStress;
mNumberOfCyclesGlobal = NumberOfCyclesGlobal;
mFatigueReductionParameter = FatigueReductionParameter;
}

ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericSmallStrainHighCycleFatigueLaw<TConstLawIntegratorType>>(*this);
}


GenericSmallStrainHighCycleFatigueLaw(const GenericSmallStrainHighCycleFatigueLaw &rOther)
: GenericSmallStrainIsotropicDamage<TConstLawIntegratorType>(rOther),
mFatigueReductionFactor(rOther.mFatigueReductionFactor),
mPreviousStresses(rOther.mPreviousStresses),
mMaxStress(rOther.mMaxStress),
mMinStress(rOther.mMinStress),
mPreviousMaxStress(rOther.mPreviousMaxStress),
mPreviousMinStress(rOther.mPreviousMinStress),
mNumberOfCyclesGlobal(rOther.mNumberOfCyclesGlobal),
mNumberOfCyclesLocal(rOther.mNumberOfCyclesLocal),
mFatigueReductionParameter(rOther.mFatigueReductionParameter),
mStressVector(rOther.mStressVector),
mMaxDetected(rOther.mMaxDetected),
mMinDetected(rOther.mMinDetected),
mWohlerStress(rOther.mWohlerStress)
{
}


~GenericSmallStrainHighCycleFatigueLaw() override
{
}




void InitializeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


bool Has(const Variable<bool>& rThisVariable) override;


bool Has(const Variable<double>& rThisVariable) override;


using ConstitutiveLaw::Has;
bool Has(const Variable<int>& rThisVariable) override;


void SetValue(
const Variable<bool>& rThisVariable,
const bool& Value,
const ProcessInfo& rCurrentProcessInfo) override;


using ConstitutiveLaw::SetValue;
void SetValue(
const Variable<int>& rThisVariable,
const int& rValue,
const ProcessInfo& rCurrentProcessInfo) override;


void SetValue(
const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo) override;


bool& GetValue(
const Variable<bool>& rThisVariable,
bool& rValue) override;


using ConstitutiveLaw::GetValue;
int& GetValue(
const Variable<int>& rThisVariable,
int& rValue) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue) override;


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Matrix& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


bool RequiresInitializeMaterialResponse() override
{
return true;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;





protected:









private:

Vector GetStressVector() {return mStressVector;}
void SetStressVector(const Vector& toStressVector) {mStressVector = toStressVector;}
double mFatigueReductionFactor = 1.0;
Vector mPreviousStresses = ZeroVector(2); 
double mMaxStress = 0.0;
double mMinStress = 0.0;
double mPreviousMaxStress = 0.0;
double mPreviousMinStress = 0.0;
unsigned int mNumberOfCyclesGlobal = 1; 
unsigned int mNumberOfCyclesLocal = 1; 
double mFatigueReductionParameter = 0.0; 
Vector mStressVector = ZeroVector(VoigtSize);
bool mMaxDetected = false; 
bool mMinDetected = false; 
double mWohlerStress = 1.0; 
double mThresholdStress = 0.0; 
double mReversionFactorRelativeError = 0.0; 
double mMaxStressRelativeError = 0.0; 
bool mNewCycleIndicator = false; 
double mCyclesToFailure = 0.0; 
double mPreviousCycleTime = 0.0; 
double mPeriod = 0.0; 






friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("FatigueReductionFactor", mFatigueReductionFactor);
rSerializer.save("PreviousStresses", mPreviousStresses);
rSerializer.save("MaxStress", mMaxStress);
rSerializer.save("MinStress", mMinStress);
rSerializer.save("PreviousMaxStress", mPreviousMaxStress);
rSerializer.save("PreviousMinStress", mPreviousMinStress);
rSerializer.save("NumberOfCyclesGlobal", mNumberOfCyclesGlobal);
rSerializer.save("NumberOfCyclesLocal", mNumberOfCyclesLocal);
rSerializer.save("FatigueReductionParameter", mFatigueReductionParameter);
rSerializer.save("StressVector", mStressVector);
rSerializer.save("MaxDetected", mMaxDetected);
rSerializer.save("MinDetected", mMinDetected);
rSerializer.save("WohlerStress", mWohlerStress);
rSerializer.save("ThresholdStress", mThresholdStress);
rSerializer.save("ReversionFactorRelativeError", mReversionFactorRelativeError);
rSerializer.save("MaxStressRelativeError", mMaxStressRelativeError);
rSerializer.save("NewCycleIndicator", mNewCycleIndicator);
rSerializer.save("CyclesToFailure", mCyclesToFailure);
rSerializer.save("PreviousCycleTime", mPreviousCycleTime);
rSerializer.save("Period", mPeriod);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("FatigueReductionFactor", mFatigueReductionFactor);
rSerializer.load("PreviousStresses", mPreviousStresses);
rSerializer.load("MaxStress", mMaxStress);
rSerializer.load("MinStress", mMinStress);
rSerializer.load("PreviousMaxStress", mPreviousMaxStress);
rSerializer.load("PreviousMinStress", mPreviousMinStress);
rSerializer.load("NumberOfCyclesGlobal", mNumberOfCyclesGlobal);
rSerializer.load("NumberOfCyclesLocal", mNumberOfCyclesLocal);
rSerializer.load("FatigueReductionParameter", mFatigueReductionParameter);
rSerializer.load("StressVector", mStressVector);
rSerializer.load("MaxDetected", mMaxDetected);
rSerializer.load("MinDetected", mMinDetected);
rSerializer.load("WohlerStress", mWohlerStress);
rSerializer.load("ThresholdStress", mThresholdStress);
rSerializer.load("ReversionFactorRelativeError", mReversionFactorRelativeError);
rSerializer.load("MaxStressRelativeError", mMaxStressRelativeError);
rSerializer.load("NewCycleIndicator", mNewCycleIndicator);
rSerializer.load("CyclesToFailure", mCyclesToFailure);
rSerializer.load("PreviousCycleTime", mPreviousCycleTime);
rSerializer.load("Period", mPeriod);
}

}; 

} 