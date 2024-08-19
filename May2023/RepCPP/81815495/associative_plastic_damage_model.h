
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"
#include "custom_constitutive/linear_plane_strain.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"
#include "custom_utilities/constitutive_law_utilities.h"
#include "constitutive_laws_application_variables.h"

namespace Kratos
{





template<class TYieldSurfaceType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) AssociativePlasticDamageModel
: public std::conditional<TYieldSurfaceType::VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type
{
public:

typedef std::size_t SizeType;

static constexpr SizeType Dimension = TYieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = TYieldSurfaceType::VoigtSize;

typedef ProcessInfo ProcessInfoType;

typedef typename std::conditional<VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type BaseType;

static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

static constexpr double tolerance = 1.0e-8;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef array_1d<double, VoigtSize> BoundedVectorType;

typedef BoundedMatrix<double, VoigtSize, VoigtSize> BoundedMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(AssociativePlasticDamageModel);

struct PlasticDamageParameters {
BoundedMatrixType ComplianceMatrixIncrement{ZeroMatrix(VoigtSize, VoigtSize)};
BoundedMatrixType ComplianceMatrix{ZeroMatrix(VoigtSize, VoigtSize)};
BoundedMatrixType ComplianceMatrixCompression{ZeroMatrix(VoigtSize, VoigtSize)};
BoundedMatrixType ConstitutiveMatrix{ZeroMatrix(VoigtSize, VoigtSize)};
BoundedMatrixType TangentTensor{ZeroMatrix(VoigtSize, VoigtSize)};
BoundedVectorType PlasticFlow{ZeroVector(VoigtSize)};
BoundedVectorType PlasticStrain{ZeroVector(VoigtSize)};
BoundedVectorType PlasticStrainIncrement{ZeroVector(VoigtSize)};
BoundedVectorType StrainVector{ZeroVector(VoigtSize)};
BoundedVectorType StressVector{ZeroVector(VoigtSize)};
double NonLinearIndicator          = 0.0; 
double PlasticConsistencyIncrement = 0.0; 
double UniaxialStress              = 0.0;
double DamageDissipation           = 0.0; 
double DamageDissipationIncrement  = 0.0; 
double PlasticDissipation          = 0.0; 
double PlasticDissipationIncrement = 0.0; 
double TotalDissipation            = 0.0; 
double CharacteristicLength        = 0.0;
double Threshold                   = 0.0;
double Slope                       = 0.0; 
double PlasticDamageProportion     = 0.5; 
};

typedef std::function<double(const double, const double, ConstitutiveLaw::Parameters& , PlasticDamageParameters &)> ResidualFunctionType;



AssociativePlasticDamageModel()
{}


~AssociativePlasticDamageModel() override
{}




ConstitutiveLaw::Pointer Clone() const override;


AssociativePlasticDamageModel(const AssociativePlasticDamageModel &rOther)
: BaseType(rOther),
mPlasticDissipation(rOther.mPlasticDissipation),
mDamageDissipation(rOther.mDamageDissipation),
mThreshold(rOther.mThreshold),
mPlasticStrain(rOther.mPlasticStrain),
mOldStrain(rOther.mOldStrain),
mComplianceMatrix(rOther.mComplianceMatrix),
mComplianceMatrixCompression(rOther.mComplianceMatrixCompression)
{
}


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue
) override;


Vector& GetValue(
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


void SetValue(
const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<Vector >& rThisVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue
) override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void CalculateMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2 (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff (ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy (ConstitutiveLaw::Parameters& rValues) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;



void CalculateTangentTensor(
ConstitutiveLaw::ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);



void CalculateElasticComplianceMatrix(
BoundedMatrixType& rConstitutiveMatrix,
const Properties& rMaterialProperties
);


void AddIfPositive(
double&  rToBeAdded,
const double Increment
)
{
if (Increment > machine_tolerance)
rToBeAdded += Increment;
}


double MacaullyBrackets(const double Number)
{
return (Number > machine_tolerance) ? Number : 0.0;
}


void CalculatePlasticDissipationIncrement(
const Properties &rMaterialProperties,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculateDamageDissipationIncrement(
const Properties &rMaterialProperties,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculateThresholdAndSlope(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculateFlowVector(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculatePlasticStrainIncrement(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculateComplianceMatrixIncrement(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculatePlasticConsistencyIncrement(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void IntegrateStressPlasticDamageMechanics(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void CalculateConstitutiveMatrix(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPlasticDamageParameters);


void UpdateInternalVariables(
PlasticDamageParameters &rPlasticDamageParameters
);


void CheckMinimumFractureEnergy(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rPDParameters
);


void InitializePlasticDamageParameters(
const BoundedVectorType& rStrainVector,
const Properties& rMaterialProperties,
const double CharateristicLength,
PlasticDamageParameters &rPlasticDamageParameters
)
{
rPlasticDamageParameters.PlasticDissipation     = mPlasticDissipation;
rPlasticDamageParameters.DamageDissipation      = mDamageDissipation;
rPlasticDamageParameters.TotalDissipation       = mPlasticDissipation + mDamageDissipation;
rPlasticDamageParameters.Threshold              = mThreshold;
noalias(rPlasticDamageParameters.PlasticStrain) = mPlasticStrain;
noalias(rPlasticDamageParameters.ComplianceMatrix) = mComplianceMatrix;
noalias(rPlasticDamageParameters.ComplianceMatrixCompression) = mComplianceMatrixCompression;
noalias(rPlasticDamageParameters.StrainVector) = rStrainVector;
rPlasticDamageParameters.CharacteristicLength  = CharateristicLength;
rPlasticDamageParameters.PlasticDamageProportion = rMaterialProperties[PLASTIC_DAMAGE_PROPORTION];
}


void CalculateAnalyticalTangentTensor(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rParam
);


void AddNonLinearDissipation(
PlasticDamageParameters &rPDParameters
)
{
rPDParameters.DamageDissipation  += rPDParameters.DamageDissipationIncrement;
rPDParameters.DamageDissipation = (rPDParameters.DamageDissipation > 0.99999) ?
0.99999 : rPDParameters.DamageDissipation;

rPDParameters.PlasticDissipation += rPDParameters.PlasticDissipationIncrement;
rPDParameters.PlasticDissipation = (rPDParameters.PlasticDissipation > 0.99999) ?
0.99999 : rPDParameters.PlasticDissipation;

rPDParameters.TotalDissipation = (rPDParameters.PlasticDissipation +
rPDParameters.DamageDissipation);
rPDParameters.TotalDissipation = (rPDParameters.TotalDissipation > 0.99999) ?
0.99999 : rPDParameters.TotalDissipation;
}


static double CalculateVolumetricFractureEnergy( 
const Properties& rMaterialProperties,
PlasticDamageParameters &rPDParameters
);


double CalculatePlasticDenominator(
ConstitutiveLaw::Parameters& rValues,
PlasticDamageParameters &rParam);


double CalculateThresholdImplicitExpression(
ResidualFunctionType &rF,
ResidualFunctionType &rdF_dk,
ConstitutiveLaw::Parameters &rValues,
PlasticDamageParameters &rPDParameters,
const double MaxThreshold = std::numeric_limits<double>::max());


double CalculateSlopeFiniteDifferences(
ResidualFunctionType &rF,
ResidualFunctionType &rdF_dk,
ConstitutiveLaw::Parameters &rValues,
PlasticDamageParameters &rPDParameters,
const double MaxThreshold = std::numeric_limits<double>::max());


ResidualFunctionType ExponentialSofteningImplicitFunction();


ResidualFunctionType ExponentialSofteningImplicitFunctionDerivative();


ResidualFunctionType ExponentialHardeningImplicitFunction();


ResidualFunctionType ExponentialHardeningImplicitFunctionDerivative();
protected:






private:



double mPlasticDissipation = 0.0;
double mDamageDissipation  = 0.0;
double mThreshold          = 0.0;
BoundedVectorType mPlasticStrain    = ZeroVector(VoigtSize);
BoundedVectorType mOldStrain        = ZeroVector(VoigtSize);
BoundedMatrixType mComplianceMatrix = ZeroMatrix(VoigtSize, VoigtSize);
BoundedMatrixType mComplianceMatrixCompression = ZeroMatrix(VoigtSize, VoigtSize);




friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("PlasticDissipation", mPlasticDissipation);
rSerializer.save("DamageDissipation", mDamageDissipation);
rSerializer.save("Threshold", mThreshold);
rSerializer.save("PlasticStrain", mPlasticStrain);
rSerializer.save("OldStrain", mOldStrain);
rSerializer.save("ComplianceMatrix", mComplianceMatrix);
rSerializer.save("ComplianceMatrixCompression", mComplianceMatrixCompression);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("PlasticDissipation", mPlasticDissipation);
rSerializer.load("DamageDissipation", mDamageDissipation);
rSerializer.load("Threshold", mThreshold);
rSerializer.load("PlasticStrain", mPlasticStrain);
rSerializer.load("OldStrain", mOldStrain);
rSerializer.load("ComplianceMatrix", mComplianceMatrix);
rSerializer.load("ComplianceMatrixCompression", mComplianceMatrixCompression);
}


}; 
}  
