
#pragma once


#include "includes/define.h"
#include "includes/checks.h"
#include "includes/properties.h"
#include "utilities/math_utils.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"
#include "constitutive_laws_application_variables.h"
#include "generic_cl_integrator_plasticity.h"

namespace Kratos
{


typedef std::size_t SizeType;




template<class TYieldSurfaceType>
class GenericConstitutiveLawIntegratorKinematicPlasticity
{
public:

static constexpr double tolerance = std::numeric_limits<double>::epsilon();

typedef std::size_t IndexType;

typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension = YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef array_1d<double, VoigtSize> BoundedArrayType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

typedef typename YieldSurfaceType::PlasticPotentialType PlasticPotentialType;

KRATOS_CLASS_POINTER_DEFINITION(GenericConstitutiveLawIntegratorKinematicPlasticity);


enum class HardeningCurveType
{
LinearSoftening = 0,
ExponentialSoftening = 1,
InitialHardeningExponentialSoftening = 2,
PerfectPlasticity = 3,
CurveFittingHardening = 4
};

enum class KinematicHardeningType
{
LinearKinematicHardening = 0,
ArmstrongFrederickKinematicHardening = 1,
AraujoVoyiadjisKinematicHardening = 2
};


GenericConstitutiveLawIntegratorKinematicPlasticity()
{
}

GenericConstitutiveLawIntegratorKinematicPlasticity(GenericConstitutiveLawIntegratorKinematicPlasticity const &rOther)
{
}

GenericConstitutiveLawIntegratorKinematicPlasticity &operator=(GenericConstitutiveLawIntegratorKinematicPlasticity const &rOther)
{
return *this;
}

virtual ~GenericConstitutiveLawIntegratorKinematicPlasticity()
{
}




static void IntegrateStressVector(
BoundedArrayType& rPredictiveStressVector,
Vector& rStrainVector,
double& rUniaxialStress,
double& rThreshold,
double& rPlasticDenominator,
BoundedArrayType& rYieldSurfaceDerivative,
BoundedArrayType& rDerivativePlasticPotential,
double& rPlasticDissipation,
BoundedArrayType& rPlasticStrainIncrement,
Matrix& rConstitutiveMatrix,
Vector& rPlasticStrain,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength,
Vector& rBackStressVector,
const Vector& rPreviousStressVector
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

bool is_converged = false;
IndexType iteration = 0, max_iter = r_material_properties.Has(MAX_NUMBER_NL_CL_ITERATIONS) ? r_material_properties.GetValue(MAX_NUMBER_NL_CL_ITERATIONS) : 100;
BoundedArrayType delta_sigma;
double plastic_consistency_factor_increment, threshold_indicator;
BoundedArrayType kin_hard_stress_vector;
Matrix tangent_tensor = ZeroMatrix(6,6);

while (is_converged == false && iteration <= max_iter) {
threshold_indicator = rUniaxialStress - rThreshold;
plastic_consistency_factor_increment = threshold_indicator * rPlasticDenominator;
noalias(rPlasticStrainIncrement) = plastic_consistency_factor_increment * rDerivativePlasticPotential;
noalias(rPlasticStrain) += rPlasticStrainIncrement;
noalias(delta_sigma) = prod(rConstitutiveMatrix, rPlasticStrainIncrement);
noalias(rPredictiveStressVector) -= delta_sigma;
CalculateBackStress(rPredictiveStressVector, rValues, rPreviousStressVector,
rPlasticStrainIncrement, rBackStressVector);
noalias(kin_hard_stress_vector) = rPredictiveStressVector - rBackStressVector;
threshold_indicator = CalculatePlasticParameters(kin_hard_stress_vector, rStrainVector, rUniaxialStress, rThreshold,
rPlasticDenominator, rYieldSurfaceDerivative, rDerivativePlasticPotential, rPlasticDissipation, rPlasticStrainIncrement,
rConstitutiveMatrix, rValues, CharacteristicLength, rPlasticStrain, rBackStressVector);


if (std::abs(threshold_indicator) <= std::abs(1.0e-4 * rThreshold)) { 
is_converged = true;
} else {
iteration++;
}
}
CalculateTangentMatrix(tangent_tensor, rConstitutiveMatrix, rYieldSurfaceDerivative, rDerivativePlasticPotential, rPlasticDenominator);
noalias(rConstitutiveMatrix) = tangent_tensor;
KRATOS_WARNING_IF("GenericConstitutiveLawIntegratorKinematicPlasticity", iteration > max_iter) << "Maximum number of iterations in plasticity loop reached..." << std::endl;
}


static void CalculateTangentMatrix(
Matrix& rTangent,
const Matrix& rElasticMatrix,
const array_1d<double, VoigtSize>& rFFluxVector,
const array_1d<double, VoigtSize>& rGFluxVector,
const double Denominator
)
{
rTangent = rElasticMatrix - outer_prod(Vector(prod(rElasticMatrix, rGFluxVector)), Vector(prod(rElasticMatrix, rFFluxVector))) * Denominator;
}



static double CalculatePlasticParameters(
BoundedArrayType& rPredictiveStressVector,
Vector& rStrainVector,
double& rUniaxialStress,
double& rThreshold,
double& rPlasticDenominator,
BoundedArrayType& rYieldSurfaceDerivative,
BoundedArrayType& rDerivativePlasticPotential,
double& rPlasticDissipation,
BoundedArrayType& rPlasticStrainIncrement,
const Matrix& rConstitutiveMatrix,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength,
const Vector& rPlasticStrain,
const Vector& rBackStressVector
)
{
BoundedArrayType deviator = ZeroVector(6);
BoundedArrayType h_capa = ZeroVector(6);
double J2, tensile_indicator_factor, compression_indicator_factor, slope, hardening_parameter, equivalent_plastic_strain;

YieldSurfaceType::CalculateEquivalentStress( rPredictiveStressVector, rStrainVector, rUniaxialStress, rValues);
const double I1 = rPredictiveStressVector[0] + rPredictiveStressVector[1] + rPredictiveStressVector[2];
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rPredictiveStressVector, I1, deviator, J2);
CalculateDerivativeYieldSurface(rPredictiveStressVector, deviator, J2, rYieldSurfaceDerivative, rValues);
CalculateDerivativePlasticPotential(rPredictiveStressVector, deviator, J2, rDerivativePlasticPotential, rValues);
CalculateIndicatorsFactors(rPredictiveStressVector, tensile_indicator_factor,compression_indicator_factor);
CalculatePlasticDissipation(rPredictiveStressVector, tensile_indicator_factor,compression_indicator_factor, rPlasticStrainIncrement,rPlasticDissipation, h_capa, rValues, CharacteristicLength);
CalculateEquivalentPlasticStrain(rPredictiveStressVector, rUniaxialStress, rPlasticStrain, tensile_indicator_factor, rValues, equivalent_plastic_strain);
CalculateEquivalentStressThreshold(rPlasticDissipation, tensile_indicator_factor,compression_indicator_factor, rThreshold, slope, rValues, equivalent_plastic_strain, CharacteristicLength);
CalculateHardeningParameter(rDerivativePlasticPotential, slope, h_capa, hardening_parameter);
CalculatePlasticDenominator(rYieldSurfaceDerivative, rDerivativePlasticPotential, rConstitutiveMatrix, hardening_parameter, rPlasticDenominator, rBackStressVector, rValues);

return rUniaxialStress - rThreshold;
}


static void CalculateDerivativeYieldSurface(
const BoundedArrayType& rPredictiveStressVector,
const BoundedArrayType& rDeviator,
const double J2,
BoundedArrayType& rDerivativeYieldSurface,
ConstitutiveLaw::Parameters& rValues
)
{
YieldSurfaceType::CalculateYieldSurfaceDerivative(rPredictiveStressVector, rDeviator, J2, rDerivativeYieldSurface, rValues);
}


static void CalculateBackStress(
BoundedArrayType& rPredictiveStressVector,
ConstitutiveLaw::Parameters& rValues,
const Vector& rPreviousStressVector,
const Vector& rPlasticStrainIncrement,
Vector& rBackStressVector
)
{
const Vector& r_kinematic_parameters = rValues.GetMaterialProperties()[KINEMATIC_PLASTICITY_PARAMETERS];
const unsigned int kinematic_hardening_type = rValues.GetMaterialProperties()[KINEMATIC_HARDENING_TYPE];

switch (static_cast<KinematicHardeningType>(kinematic_hardening_type))
{
double pDot, denominator, dot_product_dp;
case KinematicHardeningType::LinearKinematicHardening:
KRATOS_ERROR_IF(r_kinematic_parameters.size() == 0) << "Kinematic Parameters not defined..." << std::endl;
rBackStressVector += 2.0 / 3.0 * r_kinematic_parameters[0] * rPlasticStrainIncrement;
break;

case KinematicHardeningType::ArmstrongFrederickKinematicHardening:
KRATOS_ERROR_IF(r_kinematic_parameters.size() < 2) << "Kinematic Parameters not defined..." << std::endl;
dot_product_dp = 0.0;
for (IndexType i = 0; i < rPlasticStrainIncrement.size(); ++i) {
dot_product_dp += rPlasticStrainIncrement[i] * rPlasticStrainIncrement[i];
}
pDot = std::sqrt(2.0 / 3.0 * dot_product_dp);
denominator = 1.0 + (r_kinematic_parameters[1] * pDot);
rBackStressVector = (rBackStressVector + ((2.0 / 3.0 * r_kinematic_parameters[0]) * rPlasticStrainIncrement)) / denominator;
break;

case KinematicHardeningType::AraujoVoyiadjisKinematicHardening:
KRATOS_ERROR_IF(r_kinematic_parameters.size() != 3) << "Kinematic Parameters not defined..." << std::endl;
dot_product_dp = 0.0;
for (IndexType i = 0; i < rPlasticStrainIncrement.size(); ++i) {
dot_product_dp += rPlasticStrainIncrement[i] * rPlasticStrainIncrement[i];
}
pDot = std::sqrt(2.0 / 3.0 * dot_product_dp);
denominator = 1.0 + (r_kinematic_parameters[1] * pDot);
if (pDot > tolerance) {
rBackStressVector = (rBackStressVector + ((2.0 / 3.0 * r_kinematic_parameters[0]) * rPlasticStrainIncrement)) / denominator;
} else {
const Vector& r_delta_stress = rPredictiveStressVector - rPreviousStressVector;
rBackStressVector = (rBackStressVector + ((2.0 / 3.0 * r_kinematic_parameters[0]) * rPlasticStrainIncrement) +
r_kinematic_parameters[2] * r_delta_stress) / denominator;
}
break;
default:
KRATOS_ERROR << " The Kinematic hardening type of plasticity is not set or wrong..." << kinematic_hardening_type << std::endl;
break;
}
}


static void CalculateDerivativePlasticPotential(
const BoundedArrayType& rPredictiveStressVector,
const BoundedArrayType& rDeviator,
const double J2,
BoundedArrayType& rDerivativePlasticPotential,
ConstitutiveLaw::Parameters& rValues
)
{
YieldSurfaceType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rDerivativePlasticPotential, rValues);
}


static void CalculateIndicatorsFactors(
const BoundedArrayType& rPredictiveStressVector,
double& rTensileIndicatorFactor,
double& rCompressionIndicatorFactor
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateIndicatorsFactors(
rPredictiveStressVector,rTensileIndicatorFactor,rCompressionIndicatorFactor);
}


static void CalculatePlasticDissipation(
const BoundedArrayType& rPredictiveStressVector,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
const Vector& PlasticStrainInc,
double& rPlasticDissipation,
BoundedArrayType& rHCapa,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculatePlasticDissipation(
rPredictiveStressVector,TensileIndicatorFactor,CompressionIndicatorFactor,
PlasticStrainInc,rPlasticDissipation,rHCapa,rValues,CharacteristicLength);
}


static void CalculateEquivalentStressThreshold(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues,
const double EquivalentPlasticStrain,
const double CharacteristicLength
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThreshold(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,
rSlope,rValues,EquivalentPlasticStrain,CharacteristicLength);
}


static void CalculateEquivalentStressThresholdHardeningCurveLinearSoftening(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThresholdHardeningCurveLinearSoftening(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,rSlope,rValues);
}


static void CalculateEquivalentStressThresholdHardeningCurveExponentialSoftening(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThresholdHardeningCurveExponentialSoftening(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,rSlope,rValues,CharacteristicLength);
}


static void CalculateEquivalentStressThresholdHardeningCurveInitialHardeningExponentialSoftening(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThresholdHardeningCurveInitialHardeningExponentialSoftening(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,rSlope,rValues);
}


static void CalculateEquivalentStressThresholdHardeningCurvePerfectPlasticity(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThresholdHardeningCurvePerfectPlasticity(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,rSlope,rValues);
}


static void CalculateEquivalentStressThresholdCurveFittingHardening(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues,
const double EquivalentPlasticStrain,
const double CharacteristicLength
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentStressThresholdCurveFittingHardening(
PlasticDissipation,TensileIndicatorFactor,CompressionIndicatorFactor,rEquivalentStressThreshold,rSlope,rValues,
EquivalentPlasticStrain,CharacteristicLength);
}


static void CalculateEquivalentPlasticStrain(
const Vector& rStressVector,
const double UniaxialStress,
const Vector& rPlasticStrain,
const double r0,
ConstitutiveLaw::Parameters& rValues,
double& rEquivalentPlasticStrain
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateEquivalentPlasticStrain(
rStressVector,UniaxialStress,rPlasticStrain,r0,rValues,rEquivalentPlasticStrain);
}


static void GetInitialUniaxialThreshold(ConstitutiveLaw::Parameters& rValues, double& rThreshold)
{
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, rThreshold);
}


static void CalculateHardeningParameter(
const BoundedArrayType& rGFlux,
const double SlopeThreshold,
const BoundedArrayType& rHCapa,
double& rHardeningParameter
)
{
GenericConstitutiveLawIntegratorPlasticity<YieldSurfaceType>::CalculateHardeningParameter(
rGFlux,SlopeThreshold,rHCapa,rHardeningParameter);
}


static void CalculatePlasticDenominator(
const BoundedArrayType& rFFlux,
const BoundedArrayType& rGFlux,
const Matrix& rConstitutiveMatrix,
double& rHardeningParameter,
double& rPlasticDenominator,
const Vector& rBackStressVector,
ConstitutiveLaw::Parameters& rValues
)
{
const Vector& r_kinematic_parameters = rValues.GetMaterialProperties()[KINEMATIC_PLASTICITY_PARAMETERS];
const int kinematic_hardening_type = rValues.GetMaterialProperties()[KINEMATIC_HARDENING_TYPE];

const BoundedArrayType delta_vector = prod(rGFlux, rConstitutiveMatrix);
double A1 = 0.0;
for (IndexType i = 0; i < VoigtSize; ++i) {
A1 += rFFlux[i] * delta_vector[i];
}
if (r_kinematic_parameters.size() == 3) {
A1 *= (1.0 - r_kinematic_parameters[2]);
} 

double dot_fflux_gflux = 0.0, A2;
for (IndexType i = 0; i < VoigtSize; ++i) {
dot_fflux_gflux += rFFlux[i] * rGFlux[i];
}
const double two_thirds = 2.0 / 3.0;
double dot_fflux_backstress = 0.0, dot_gflux_gflux = 0.0;
switch (static_cast<KinematicHardeningType>(kinematic_hardening_type))
{
case KinematicHardeningType::LinearKinematicHardening:
A2 = two_thirds * r_kinematic_parameters[0] * dot_fflux_gflux;
break;

case KinematicHardeningType::ArmstrongFrederickKinematicHardening:
A2 = two_thirds * r_kinematic_parameters[0] * dot_fflux_gflux;
for (IndexType i = 0; i < VoigtSize; ++i) {
dot_fflux_backstress += rFFlux[i] * rBackStressVector[i];
}
for (IndexType i = 0; i < VoigtSize; ++i) {
dot_gflux_gflux += rGFlux[i] * rGFlux[i];
}
A2 -= r_kinematic_parameters[1] * dot_fflux_backstress * std::sqrt(two_thirds * dot_gflux_gflux);
break;

case KinematicHardeningType::AraujoVoyiadjisKinematicHardening:
A2 = two_thirds * r_kinematic_parameters[0] * dot_fflux_gflux;
for (IndexType i = 0; i < VoigtSize; ++i) {
dot_fflux_backstress += rFFlux[i] * rBackStressVector[i];
}
for (IndexType i = 0; i < VoigtSize; ++i) {
dot_gflux_gflux += rGFlux[i] * rGFlux[i];
}
A2 -= r_kinematic_parameters[1] * dot_fflux_backstress * std::sqrt(two_thirds * dot_gflux_gflux);
break;

default:
KRATOS_ERROR << " The Kinematic hardening type of plasticity is not set or wrong..." << kinematic_hardening_type << std::endl;
break;
}

const double A3 = rHardeningParameter;
rPlasticDenominator = 1.0 / (A1 + A2 + A3);

if (r_kinematic_parameters.size() == 3) {
rPlasticDenominator *= (1.0 - r_kinematic_parameters[2]);
} 
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YOUNG_MODULUS)) << "HARDENING_CURVE is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(HARDENING_CURVE)) << "HARDENING_CURVE is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRACTURE_ENERGY)) << "FRACTURE_ENERGY is not a defined value" << std::endl;

const int curve_type = rMaterialProperties[HARDENING_CURVE];
if (static_cast<HardeningCurveType>(curve_type) == HardeningCurveType::InitialHardeningExponentialSoftening) {
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(MAXIMUM_STRESS)) << "MAXIMUM_STRESS is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(MAXIMUM_STRESS_POSITION)) << "MAXIMUM_STRESS_POSITION is not a defined value" << std::endl;
} else if (static_cast<HardeningCurveType>(curve_type) == HardeningCurveType::CurveFittingHardening) {
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(CURVE_FITTING_PARAMETERS)) << "CURVE_FITTING_PARAMETERS is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(PLASTIC_STRAIN_INDICATORS)) << "PLASTIC_STRAIN_INDICATORS is not a defined value" << std::endl;
}

if (!rMaterialProperties.Has(YIELD_STRESS)) {
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YIELD_STRESS_TENSION)) << "YIELD_STRESS_TENSION is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YIELD_STRESS_COMPRESSION)) << "YIELD_STRESS_COMPRESSION is not a defined value" << std::endl;

const double yield_compression = rMaterialProperties[YIELD_STRESS_COMPRESSION];
const double yield_tension = rMaterialProperties[YIELD_STRESS_TENSION];

KRATOS_ERROR_IF(yield_compression < tolerance) << "Yield stress in compression almost zero or negative, include YIELD_STRESS_COMPRESSION in definition";
KRATOS_ERROR_IF(yield_tension < tolerance) << "Yield stress in tension almost zero or negative, include YIELD_STRESS_TENSION in definition";
} else {
const double yield_stress = rMaterialProperties[YIELD_STRESS];

KRATOS_ERROR_IF(yield_stress < tolerance) << "Yield stress almost zero or negative, include YIELD_STRESS in definition";
}

return TYieldSurfaceType::Check(rMaterialProperties);
}






protected:








private:








}; 





} 
