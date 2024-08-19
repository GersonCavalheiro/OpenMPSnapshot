
#pragma once



#include "custom_constitutive/composites/rule_of_mixtures_law.h"


namespace Kratos
{





template<unsigned int TDim>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) TractionSeparationLaw3D
: public ParallelRuleOfMixturesLaw<TDim>
{
public:

typedef ParallelRuleOfMixturesLaw<TDim> BaseType;

typedef std::size_t SizeType;

static constexpr SizeType VoigtSize = (TDim == 3) ? 6 : 3;

static constexpr SizeType Dimension = TDim;

static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(TractionSeparationLaw3D);



TractionSeparationLaw3D();


TractionSeparationLaw3D(const std::vector<double> &rCombinationFactors);


TractionSeparationLaw3D(const TractionSeparationLaw3D &rOther);


~TractionSeparationLaw3D() override;




ConstitutiveLaw::Pointer Clone() const override;


ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;


bool Has(const Variable<Vector>& rThisVariable) override;


Vector& GetValue(
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


bool ValidateInput(const Properties& rMaterialProperties) override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const ConstitutiveLaw::GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void CalculateMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


double CalculateDelaminationDamageExponentialSoftening (
ConstitutiveLaw::Parameters& rValues,
const double GI,
const double E,
const double T0,
const double equivalent_stress);


int Check(
const Properties& rMaterialProperties,
const ConstitutiveLaw::GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


void CalculateTangentTensor(
ConstitutiveLaw::Parameters& rValues,
const ConstitutiveLaw::StressMeasure& rStressMeasure
);

protected:






private:



Vector mDelaminationDamageModeOne;
Vector mDelaminationDamageModeTwo;
Vector mThresholdModeOne;
Vector mThresholdModeTwo;




friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ParallelRuleOfMixturesLaw<TDim> )
rSerializer.save("DelaminationDamageModeOne", mDelaminationDamageModeOne);
rSerializer.save("DelaminationDamageModeTwo", mDelaminationDamageModeTwo);
rSerializer.save("ThresholdModeOne", mThresholdModeOne);
rSerializer.save("ThresholdModeTwo", mThresholdModeTwo);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ParallelRuleOfMixturesLaw<TDim>)
rSerializer.load("DelaminationDamageModeOne", mDelaminationDamageModeOne);
rSerializer.load("DelaminationDamageModeTwo", mDelaminationDamageModeTwo);
rSerializer.load("ThresholdModeOne", mThresholdModeOne);
rSerializer.load("ThresholdModeTwo", mThresholdModeTwo);
}

}; 
}  
