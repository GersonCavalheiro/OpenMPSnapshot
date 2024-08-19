#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"

namespace Kratos
{


typedef std::size_t SizeType;





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) DamageDPlusDMinusMasonry3DLaw
: public ElasticIsotropic3D
{
public:


static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

typedef ElasticIsotropic3D BaseType;


KRATOS_CLASS_POINTER_DEFINITION(DamageDPlusDMinusMasonry3DLaw);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();

struct DamageParameters {
double DamageTension = 0.0;
double DamageCompression = 0.0;
double ThresholdTension = 0.0;
double ThresholdCompression = 0.0;
array_1d<double, VoigtSize> TensionStressVector;
array_1d<double, VoigtSize> CompressionStressVector;
double UniaxialTensionStress = 0.0;
double UniaxialCompressionStress = 0.0;
};


DamageDPlusDMinusMasonry3DLaw();



ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<DamageDPlusDMinusMasonry3DLaw>(*this);
}

SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};


DamageDPlusDMinusMasonry3DLaw(const DamageDPlusDMinusMasonry3DLaw &rOther)
: BaseType(rOther),
mTensionDamage(rOther.mTensionDamage),
mTensionThreshold(rOther.mTensionThreshold),
mNonConvTensionDamage(rOther.mNonConvTensionDamage),
mNonConvTensionThreshold(rOther.mNonConvTensionThreshold),
mCompressionDamage(rOther.mCompressionDamage),
mCompressionThreshold(rOther.mCompressionThreshold),
mNonConvCompressionDamage(rOther.mNonConvCompressionDamage),
mNonConvCompressionThreshold(rOther.mNonConvCompressionThreshold)
{
}


~DamageDPlusDMinusMasonry3DLaw() override
{
}


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


bool IntegrateStressTensionIfNecessary(
const double F_tension,
DamageParameters& Parameters,
array_1d<double, VoigtSize>& IntegratedStressVectorTension,
const array_1d<double, VoigtSize> rIntegratedStressVector,
ConstitutiveLaw::Parameters& rValues
);


bool IntegrateStressCompressionIfNecessary(
const double F_compression,
DamageParameters& Parameters,
array_1d<double, VoigtSize>& IntegratedStressVectorCompression,
array_1d<double, VoigtSize> rIntegratedStressVector,
ConstitutiveLaw::Parameters& rValues
);


void CalculateIntegratedStressVector(
Vector& IntegratedStressVectorTension,
const DamageParameters& Parameters,
ConstitutiveLaw::Parameters& rValues
);


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void FinalizeSolutionStep(
const Properties &rMaterialProperties,
const GeometryType &rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo
) override;


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






protected:



double& GetTensionThreshold() { return mTensionThreshold; }
double& GetTensionDamage() { return mTensionDamage; }
double& GetNonConvTensionThreshold() { return mNonConvTensionThreshold; }
double& GetNonConvTensionDamage() { return mNonConvTensionDamage; }

void SetTensionThreshold(const double toThreshold) { mTensionThreshold = toThreshold; }
void SetTensionDamage(const double toDamage) { mTensionDamage = toDamage; }
void SetNonConvTensionThreshold(const double toThreshold) { mNonConvTensionThreshold = toThreshold; }
void SetNonConvTensionDamage(const double toDamage) { mNonConvTensionDamage = toDamage; }

double& GetCompressionThreshold() { return mCompressionThreshold; }
double& GetCompressionDamage() { return mCompressionDamage; }
double& GetNonConvCompressionThreshold() { return mNonConvCompressionThreshold; }
double& GetNonConvCompressionDamage() { return mNonConvCompressionDamage; }

void SetCompressionThreshold(const double toThreshold) { mCompressionThreshold = toThreshold; }
void SetCompressionDamage(const double toDamage) { mCompressionDamage = toDamage; }
void SetNonConvCompressionThreshold(const double toThreshold) { mNonConvCompressionThreshold = toThreshold; }
void SetNonConvCompressionDamage(const double toDamage) { mNonConvCompressionDamage = toDamage; }

void SetTensionStress(const double toS){mTensionUniaxialStress = toS;}
void SetCompressionStress(const double toS){mCompressionUniaxialStress = toS;}





private:


double mTensionDamage = 0.0;
double mTensionThreshold = 0.0;

double mNonConvTensionDamage = 0.0;
double mNonConvTensionThreshold = 0.0;

double mCompressionDamage = 0.0;
double mCompressionThreshold = 0.0;

double mNonConvCompressionDamage = 0.0;
double mNonConvCompressionThreshold = 0.0;

double mTensionUniaxialStress = 0.0;
double mCompressionUniaxialStress = 0.0;



void CalculateTangentTensor(ConstitutiveLaw::Parameters &rValues);


void CalculateSecantTensor(ConstitutiveLaw::Parameters& rValues, Matrix& rSecantTensor);



friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("TensionDamage", mTensionDamage);
rSerializer.save("TensionThreshold", mTensionThreshold);
rSerializer.save("NonConvTensionDamage", mNonConvTensionDamage);
rSerializer.save("NonConvTensionThreshold", mNonConvTensionThreshold);
rSerializer.save("CompressionDamage", mCompressionDamage);
rSerializer.save("CompressionThreshold", mCompressionThreshold);
rSerializer.save("NonConvCompressionnDamage", mNonConvCompressionDamage);
rSerializer.save("NonConvCompressionThreshold", mNonConvCompressionThreshold);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("TensionDamage", mTensionDamage);
rSerializer.load("TensionThreshold", mTensionThreshold);
rSerializer.load("NonConvTensionDamage", mNonConvTensionDamage);
rSerializer.load("NonConvTensionThreshold", mNonConvTensionThreshold);
rSerializer.load("CompressionDamage", mCompressionDamage);
rSerializer.load("CompressionThreshold", mCompressionThreshold);
rSerializer.load("NonConvCompressionnDamage", mNonConvCompressionDamage);
rSerializer.load("NonConvCompressionThreshold", mNonConvCompressionThreshold);
}


void CalculateEquivalentStressTension(
array_1d<double, VoigtSize>& rPredictiveStressVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues);


void CalculateEquivalentStressCompression(
array_1d<double, VoigtSize>& rPredictiveStressVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues);


void IntegrateStressVectorTension(
array_1d<double,VoigtSize>& rPredictiveStressVector,
const double UniaxialStress,
double& rDamage,
double& rThreshold,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength);


void CalculateDamageParameterTension(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength);



void CalculateExponentialDamageTension(
const double UniaxialStress,
const double Threshold,
const double DamageParameter,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues,
double& rDamage);


void IntegrateStressVectorCompression(
array_1d<double,VoigtSize>& rPredictiveStressVector,
const double UniaxialStress,
double& rDamage,
double& rThreshold,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength);



void CalculateBezier3DamageCompression(
const double UniaxialStress,
double& rDamage,
double& rThreshold,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues);


void RegulateBezierDeterminators(
const double specific_dissipated_fracture_energy,
const double sp, const double sk, const double sr, const double ep,
double& ej, double& ek, double& er, double& eu);


void ComputeBezierEnergy(
double& BezierG,
const double x1, const double x2, const double x3,
const double y1, const double y2, const double y3);


double EvaluateBezierCurve(
const double Xi,
const double x1, double x2, const double x3,
const double y1, const double y2, const double y3);








}; 
}

