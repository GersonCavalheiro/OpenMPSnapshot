
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"
#include "custom_constitutive/linear_plane_strain.h"

namespace Kratos
{


typedef std::size_t SizeType;





template <class TPlasticityIntegratorType, class TDamageIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainPlasticDamageModel
: public std::conditional<TPlasticityIntegratorType::VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type
{
public:

typedef std::size_t IndexType;

static constexpr SizeType Dimension = TPlasticityIntegratorType::Dimension;

static constexpr SizeType VoigtSize = TPlasticityIntegratorType::VoigtSize;

typedef typename std::conditional<VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type BaseType;

typedef array_1d<double, VoigtSize> BoundedArrayType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainPlasticDamageModel);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();

struct PlasticDamageParameters {
array_1d<double, VoigtSize> PlasticityFFLux;
array_1d<double, VoigtSize> PlasticityGFLux;
array_1d<double, VoigtSize> DamageYieldFLux;
double DamageIndicator = 0.0;
double PlasticityIndicator = 0.0;
array_1d<double, VoigtSize> PlasticStrain;
array_1d<double, VoigtSize> StrainVector;
array_1d<double, VoigtSize> StressVector;
double DamageIncrement = 0.0;
double PlasticConsistencyIncrement = 0.0;
double UniaxialStressPlasticity = 0.0;
double UniaxialStressDamage = 0.0;
double HardeningParameterDamage = 0.0;
double DamageDissipationIncrement = 0.0;
array_1d<double, VoigtSize> PlasticStrainIncrement;
double CharacteristicLength = 0.0;
double Damage = 0.0;
double PlasticDissipation = 0.0;
double DamageDissipation = 0.0;
double DamageThreshold = 0.0;
double PlasticityThreshold = 0.0;
double PlasticDenominator = 0.0;
double UndamagedFreeEnergy = 0.0;
};


GenericSmallStrainPlasticDamageModel()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericSmallStrainPlasticDamageModel<TPlasticityIntegratorType, TDamageIntegratorType>>(*this);
}


GenericSmallStrainPlasticDamageModel(const GenericSmallStrainPlasticDamageModel &rOther)
: BaseType(rOther),
mPlasticDissipation(rOther.mPlasticDissipation),
mThresholdPlasticity(rOther.mThresholdPlasticity),
mPlasticStrain(rOther.mPlasticStrain),
mThresholdDamage(rOther.mThresholdDamage),
mDamage(rOther.mDamage)
{
}


~GenericSmallStrainPlasticDamageModel() override
{
}




void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void InitializeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;

void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


bool Has(const Variable<double> &rThisVariable) override;


bool Has(const Variable<Vector> &rThisVariable) override;


bool Has(const Variable<Matrix> &rThisVariable) override;


void SetValue(
const Variable<double> &rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<Vector> &rThisVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


double& GetValue(
const Variable<double> &rThisVariable,
double& rValue
) override;


Vector& GetValue(
const Variable<Vector> &rThisVariable,
Vector& rValue
) override;


Matrix& GetValue(
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


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


double CalculateDamageParameters(
PlasticDamageParameters& rParameters,
const Matrix& rElasticMatrix,
ConstitutiveLaw::Parameters& rValues);


void CalculateIndicatorsFactors(
const array_1d<double, 6>& rPredictiveStressVector,
double& rTensileIndicatorFactor,
double& rCompressionIndicatorFactor,
double& rSumPrincipalStresses,
array_1d<double, 3>& rPrincipalStresses);


void CheckInternalVariable(
double& rInternalVariable);


void CalculateIncrementsPlasticDamageCase(
PlasticDamageParameters& rParameters,
const Matrix& rElasticMatrix);


double CalculatePlasticParameters(
PlasticDamageParameters& rParameters,
const Matrix& rConstitutiveMatrix,
ConstitutiveLaw::Parameters& rValues);


void CalculatePlasticDenominator(
const array_1d<double, VoigtSize>& rFFlux,
const array_1d<double, VoigtSize>& rGFlux,
const Matrix& rConstitutiveMatrix,
double& rHardeningParameter,
const double Damage,
double& rPlasticDenominator);






protected:



double& GetThresholdPlasticity() { return mThresholdPlasticity; }
double& GetPlasticDissipation() { return mPlasticDissipation; }
Vector& GetPlasticStrain() { return mPlasticStrain; }

void SetThresholdPlasticity(const double ThresholdPlasticity) { mThresholdPlasticity = ThresholdPlasticity; }
void SetPlasticDissipation(const double PlasticDissipation) { mPlasticDissipation = PlasticDissipation; }
void SetPlasticStrain(const array_1d<double, VoigtSize>& rPlasticStrain) { mPlasticStrain = rPlasticStrain; }





private:


double mPlasticDissipation = 0.0;
double mThresholdPlasticity = 0.0;
Vector mPlasticStrain = ZeroVector(VoigtSize);
double mThresholdDamage = 0.0;
double mDamage = 0.0;
double mDamageDissipation = 0.0;

double mUniaxialStress = 0.0;




void CalculateTangentTensor(ConstitutiveLaw::Parameters &rValues);





friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("PlasticDissipation", mPlasticDissipation);
rSerializer.save("ThresholdPlasticity", mThresholdPlasticity);
rSerializer.save("PlasticStrain", mPlasticStrain);
rSerializer.save("ThresholdDamage", mThresholdDamage);
rSerializer.save("Damage", mDamage);
rSerializer.save("DamageDissipation", mDamageDissipation);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("PlasticDissipation", mPlasticDissipation);
rSerializer.load("ThresholdPlasticity", mThresholdPlasticity);
rSerializer.load("PlasticStrain", mPlasticStrain);
rSerializer.load("ThresholdDamage", mThresholdDamage);
rSerializer.load("Damage", mDamage);
rSerializer.load("DamageDissipation", mDamageDissipation);
}


}; 

} 
