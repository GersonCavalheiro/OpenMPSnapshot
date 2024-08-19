#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{

class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) DamageDPlusDMinusMasonry2DLaw
: public ConstitutiveLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(DamageDPlusDMinusMasonry2DLaw);


static constexpr SizeType Dimension = 2;

static constexpr SizeType VoigtSize = 3;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();



DamageDPlusDMinusMasonry2DLaw();


~DamageDPlusDMinusMasonry2DLaw() override
{
}


ConstitutiveLaw::Pointer Clone() const override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};

struct CalculationData{

double YoungModulus;								double PoissonRatio;
Matrix ElasticityMatrix;

double YieldStressTension;							double FractureEnergyTension;

double DamageOnsetStressCompression;				double YieldStressCompression;
double ResidualStressCompression;					double YieldStrainCompression;
double BezierControllerC1;							double BezierControllerC2;
double BezierControllerC3;							double FractureEnergyCompression;
double BiaxialCompressionMultiplier;				double ShearCompressionReductor;

array_1d<double,3> EffectiveStressVector;			array_1d<double,2> PrincipalStressVector;
array_1d<double,3> EffectiveTensionStressVector;	array_1d<double,3> EffectiveCompressionStressVector;
Matrix ProjectionTensorTension;						Matrix ProjectionTensorCompression;

double CharacteristicLength;						double DeltaTime;
int TensionYieldModel;

};


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


bool Has(const Variable<Matrix>& rThisVariable) override;


bool Has(const Variable<array_1d<double, 3 > >& rThisVariable) override;


bool Has(const Variable<array_1d<double, 6 > >& rThisVariable) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& GetValue(
const Variable<Vector>& rThisVariable,
Vector& rValue) override;


Matrix& GetValue(
const Variable<Matrix>& rThisVariable,
Matrix& rValue) override;


array_1d<double, 3 > & GetValue(
const Variable<array_1d<double, 3 > >& rVariable,
array_1d<double, 3 > & rValue) override;


array_1d<double, 6 > & GetValue(
const Variable<array_1d<double, 6 > >& rVariable,
array_1d<double, 6 > & rValue) override;


void SetValue(
const Variable<double>& rVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo) override;


void SetValue(
const Variable<Vector >& rVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo) override;


void SetValue(
const Variable<Matrix >& rVariable,
const Matrix& rValue,
const ProcessInfo& rCurrentProcessInfo) override;


void SetValue(
const Variable<array_1d<double, 3 > >& rVariable,
const array_1d<double, 3 > & rValue,
const ProcessInfo& rCurrentProcessInfo) override;


void SetValue(
const Variable<array_1d<double, 6 > >& rVariable,
const array_1d<double, 6 > & rValue,
const ProcessInfo& rCurrentProcessInfo) override;


bool ValidateInput(const Properties& rMaterialProperties) override;


StrainMeasure GetStrainMeasure() override;


StressMeasure GetStressMeasure() override;


bool IsIncremental() override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;


void InitializeMaterialResponsePK2 (
ConstitutiveLaw::Parameters& rValues) override;


void InitializeSolutionStep(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void ResetMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;


void GetLawFeatures(Features& rFeatures) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo) const override;


void CalculateMaterialResponse(const Vector& StrainVector,
const Matrix& DeformationGradient,
Vector& StressVector,
Matrix& AlgorithmicTangent,
const ProcessInfo& rCurrentProcessInfo,
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
bool CalculateStresses = true,
int CalculateTangent = true,
bool SaveInternalVariables = true);

protected:


bool InitializeDamageLaw = false;


double PreviousThresholdTension = 0.0;			double PreviousThresholdCompression = 0.0;		    

double CurrentThresholdTension = 0.0;     		double CurrentThresholdCompression = 0.0;		    
double ThresholdTension        = 0.0;			double ThresholdCompression        = 0.0;			

double DamageParameterTension = 0.0;			double DamageParameterCompression = 0.0;
double UniaxialStressTension  = 0.0;			double UniaxialStressCompression  = 0.0;

double InitialCharacteristicLength = 0.0;

double CurrentDeltaTime  = 0.0;					
double PreviousDeltaTime = 0.0;					

double TemporaryImplicitThresholdTension = 0.0;
double TemporaryImplicitThresholdTCompression = 0.0;





void InitializeCalculationData(
const Properties& props,
const GeometryType& geom,
const ProcessInfo& pinfo,
CalculationData& data);


void CalculateElasticityMatrix(
CalculationData& data);


void TensionCompressionSplit(
CalculationData& data);


void ConstructProjectionTensors(
CalculationData& data);


void CalculateEquivalentStressTension(
CalculationData& data,
double& UniaxialStressTension);


void CalculateEquivalentStressCompression(
CalculationData& data,
double& UniaxialStressCompression);


void CalculateDamageTension(
CalculationData& data,
double internal_variable,
double& rDamageTension);




void CalculateDamageCompression(
CalculationData& data,
double internal_variable,
double& rDamage);


void ComputeBezierEnergy(
double& rBezierEnergy,
double& rBezierEnergy1,
double s_p, double s_k, double s_r,
double e_p, double e_j, double e_k, double e_r, double e_u);


double EvaluateBezierArea(
double x1, double x2, double x3,
double y1, double y2, double y3);


void ApplyBezierStretcherToStrains(
double stretcher, double e_p, double& e_j,
double& e_k, double& e_r, double& e_u);


void EvaluateBezierCurve(
double& rDamageParameter, double xi,
double x1, double x2, double x3,
double y1, double y2, double y3);


void ComputeCharacteristicLength(
const GeometryType& geom,
double& rCharacteristicLength);


void CalculateMaterialResponseInternal(
const Vector& strain_vector,
Vector& stress_vector,
CalculationData& data,
const Properties props);


void CheckDamageLoadingUnloading(
bool& is_damaging_tension,
bool& is_damaging_compression);


void CalculateTangentTensor(
Parameters& rValues,
Vector strain_vector,
Vector stress_vector,
CalculationData& data,
const Properties& props);


void CalculateSecantTensor(
Parameters& rValues,
CalculationData& data);





private:







friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw );
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw );
}


}; 

} 