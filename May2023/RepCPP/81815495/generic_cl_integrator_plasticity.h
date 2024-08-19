
#pragma once


#include "includes/define.h"
#include "includes/checks.h"
#include "includes/properties.h"
#include "utilities/math_utils.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"
#include "constitutive_laws_application_variables.h"

namespace Kratos
{


typedef std::size_t SizeType;




template<class TYieldSurfaceType>
class GenericConstitutiveLawIntegratorPlasticity
{
public:

static constexpr double tolerance = std::numeric_limits<double>::epsilon();

typedef std::size_t IndexType;

typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension = YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef typename YieldSurfaceType::PlasticPotentialType PlasticPotentialType;

KRATOS_CLASS_POINTER_DEFINITION(GenericConstitutiveLawIntegratorPlasticity);


enum class HardeningCurveType
{
LinearSoftening = 0,
ExponentialSoftening = 1,
InitialHardeningExponentialSoftening = 2,
PerfectPlasticity = 3,
CurveFittingHardening = 4,
LinearExponentialSoftening = 5,
CurveDefinedByPoints = 6
};


GenericConstitutiveLawIntegratorPlasticity()
{
}

GenericConstitutiveLawIntegratorPlasticity(GenericConstitutiveLawIntegratorPlasticity const &rOther)
{
}

GenericConstitutiveLawIntegratorPlasticity &operator=(GenericConstitutiveLawIntegratorPlasticity const &rOther)
{
return *this;
}

virtual ~GenericConstitutiveLawIntegratorPlasticity()
{
}




static void IntegrateStressVector(
array_1d<double, VoigtSize>& rPredictiveStressVector,
Vector& rStrainVector,
double& rUniaxialStress,
double& rThreshold,
double& rPlasticDenominator,
array_1d<double, VoigtSize>& rFflux,
array_1d<double, VoigtSize>& rGflux,
double& rPlasticDissipation,
array_1d<double, VoigtSize>& rPlasticStrainIncrement,
Matrix& rConstitutiveMatrix,
Vector& rPlasticStrain,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

bool is_converged = false;
IndexType iteration = 0, max_iter = r_material_properties.Has(MAX_NUMBER_NL_CL_ITERATIONS)
? r_material_properties.GetValue(MAX_NUMBER_NL_CL_ITERATIONS) : 100;
array_1d<double, VoigtSize> delta_sigma;
double plastic_consistency_factor_increment;
double F = rUniaxialStress - rThreshold;
Matrix tangent_tensor = ZeroMatrix(6,6);

while (is_converged == false && iteration <= max_iter) {
plastic_consistency_factor_increment = F * rPlasticDenominator;
if (plastic_consistency_factor_increment < 0.0) plastic_consistency_factor_increment = 0.0; 
noalias(rPlasticStrainIncrement) = plastic_consistency_factor_increment * rGflux;
noalias(rPlasticStrain) += rPlasticStrainIncrement;
noalias(delta_sigma) = prod(rConstitutiveMatrix, rPlasticStrainIncrement);

noalias(rPredictiveStressVector) -= delta_sigma;

F = CalculatePlasticParameters(rPredictiveStressVector, rStrainVector, rUniaxialStress, rThreshold,
rPlasticDenominator, rFflux, rGflux, rPlasticDissipation, rPlasticStrainIncrement,
rConstitutiveMatrix, rValues, CharacteristicLength, rPlasticStrain);

if (F <= std::abs(1.0e-4 * rThreshold)) { 
is_converged = true;
} else {
iteration++;
}
}
CalculateTangentMatrix(tangent_tensor, rConstitutiveMatrix, rFflux, rGflux, rPlasticDenominator);
noalias(rConstitutiveMatrix) = tangent_tensor;
KRATOS_WARNING_IF("GenericConstitutiveLawIntegratorPlasticity", iteration > max_iter) << "Maximum number of iterations in plasticity loop reached..." << std::endl;
}


static double CalculatePlasticParameters(
array_1d<double, VoigtSize>& rPredictiveStressVector,
Vector& rStrainVector,
double& rUniaxialStress,
double& rThreshold,
double& rPlasticDenominator,
array_1d<double, VoigtSize>& rFflux,
array_1d<double, VoigtSize>& rGflux,
double& rPlasticDissipation,
array_1d<double, VoigtSize>& rPlasticStrainIncrement,
const Matrix& rConstitutiveMatrix,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength,
const Vector& rPlasticStrain
)
{
array_1d<double, VoigtSize> deviator = ZeroVector(6);
array_1d<double, VoigtSize> h_capa = ZeroVector(6);
double J2, tensile_indicator_factor, compression_indicator_factor, slope, hardening_parameter, equivalent_plastic_strain;

YieldSurfaceType::CalculateEquivalentStress( rPredictiveStressVector, rStrainVector, rUniaxialStress, rValues);
const double I1 = rPredictiveStressVector[0] + rPredictiveStressVector[1] + rPredictiveStressVector[2];
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rPredictiveStressVector, I1, deviator, J2);
CalculateFFluxVector(rPredictiveStressVector, deviator, J2, rFflux, rValues);
CalculateGFluxVector(rPredictiveStressVector, deviator, J2, rGflux, rValues);
CalculateIndicatorsFactors(rPredictiveStressVector, tensile_indicator_factor,compression_indicator_factor);
CalculatePlasticDissipation(rPredictiveStressVector, tensile_indicator_factor,compression_indicator_factor, rPlasticStrainIncrement,rPlasticDissipation, h_capa, rValues, CharacteristicLength);
CalculateEquivalentPlasticStrain(rPredictiveStressVector, rUniaxialStress, rPlasticStrain, tensile_indicator_factor, rValues, equivalent_plastic_strain);
CalculateEquivalentStressThreshold(rPlasticDissipation, tensile_indicator_factor,compression_indicator_factor, rThreshold, slope, rValues, equivalent_plastic_strain, CharacteristicLength);
CalculateHardeningParameter(rGflux, slope, h_capa, hardening_parameter);
CalculatePlasticDenominator(rFflux, rGflux, rConstitutiveMatrix, hardening_parameter, rPlasticDenominator);

return rUniaxialStress - rThreshold;
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



static void CalculateFFluxVector(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rFFluxVector,
ConstitutiveLaw::Parameters& rValues
)
{
YieldSurfaceType::CalculateYieldSurfaceDerivative(rPredictiveStressVector, rDeviator, J2, rFFluxVector, rValues);
}


static void CalculateGFluxVector(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFluxVector,
ConstitutiveLaw::Parameters& rValues
)
{
YieldSurfaceType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rGFluxVector, rValues);
}


static void CalculateIndicatorsFactors(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
double& rTensileIndicatorFactor,
double& rCompressionIndicatorFactor
)
{
if (norm_2(rPredictiveStressVector) < 1.0e-8) {
rTensileIndicatorFactor = 1.0;
rCompressionIndicatorFactor = 0.0;
return;
}

array_1d<double, Dimension> principal_stresses = ZeroVector(Dimension);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculatePrincipalStresses(principal_stresses, rPredictiveStressVector);

double suma = 0.0, sumb = 0.0, sumc = 0.0;
double aux_sa;

for (IndexType i = 0; i < Dimension; ++i) {
aux_sa = std::abs(principal_stresses[i]);
suma += aux_sa;
sumb += 0.5 * (principal_stresses[i] + aux_sa);
sumc += 0.5 * (-principal_stresses[i] + aux_sa);
}

if (std::abs(suma) > tolerance) {
rTensileIndicatorFactor = sumb / suma;
rCompressionIndicatorFactor = sumc / suma;
} else {
rTensileIndicatorFactor = sumb;
rCompressionIndicatorFactor = sumc;
}

if ((std::abs(rTensileIndicatorFactor) + std::abs(rCompressionIndicatorFactor)) < tolerance) {
rTensileIndicatorFactor = 0.0;
rCompressionIndicatorFactor = 0.0;
return;
}
}


static void CalculatePlasticDissipation(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
const Vector& PlasticStrainInc,
double& rPlasticDissipation,
array_1d<double, VoigtSize>& rHCapa,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double young_modulus = r_material_properties[YOUNG_MODULUS];
const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;
const double fracture_energy_tension = r_material_properties[FRACTURE_ENERGY]; 
const double fracture_energy_compression = r_material_properties[FRACTURE_ENERGY] * std::pow(n, 2); 

const double characteristic_fracture_energy_tension = fracture_energy_tension / CharacteristicLength;
const double characteristic_fracture_energy_compression = fracture_energy_compression / CharacteristicLength;

const double hlim = 2.0 * young_modulus * fracture_energy_compression / (std::pow(yield_compression, 2));
KRATOS_ERROR_IF(CharacteristicLength > hlim) << "The Fracture Energy is to low: " << characteristic_fracture_energy_compression << std::endl;

double constant0 = 0.0, constant1 = 0.0, dplastic_dissipation = 0.0;
if (characteristic_fracture_energy_tension > 0.000001) {
constant0 = TensileIndicatorFactor / characteristic_fracture_energy_tension;
constant1 = CompressionIndicatorFactor / characteristic_fracture_energy_compression;
}
const double constant = constant0 + constant1;

for (IndexType i = 0; i < VoigtSize; ++i) {
rHCapa[i] = constant * rPredictiveStressVector[i];
dplastic_dissipation += rHCapa[i] * PlasticStrainInc[i];
}

if (dplastic_dissipation < 0.0 || dplastic_dissipation > 1.0)
dplastic_dissipation = 0.0;

rPlasticDissipation += dplastic_dissipation;
if (rPlasticDissipation >= 0.9999)
rPlasticDissipation = 0.9999;
else if (rPlasticDissipation < 0.0)
rPlasticDissipation = 0.0;

KRATOS_DEBUG_ERROR_IF(std::isnan(rPlasticDissipation)) << "rPlasticDissipation is nan" << std::endl;
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
const Properties& r_material_properties = rValues.GetMaterialProperties();
const int curve_type = r_material_properties[HARDENING_CURVE];
BoundedVector<double, 2> slopes, eq_thresholds;

for (IndexType i = 0; i < 2; ++i) { 
switch (static_cast<HardeningCurveType>(curve_type))
{
case HardeningCurveType::LinearSoftening:
CalculateEquivalentStressThresholdHardeningCurveLinearSoftening(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues);
break;

case HardeningCurveType::ExponentialSoftening:
CalculateEquivalentStressThresholdHardeningCurveExponentialSoftening(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues, CharacteristicLength);
break;

case HardeningCurveType::InitialHardeningExponentialSoftening:
CalculateEquivalentStressThresholdHardeningCurveInitialHardeningExponentialSoftening(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues);
break;

case HardeningCurveType::PerfectPlasticity:
CalculateEquivalentStressThresholdHardeningCurvePerfectPlasticity(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues);
break;

case HardeningCurveType::CurveFittingHardening: 
CalculateEquivalentStressThresholdCurveFittingHardening(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues, EquivalentPlasticStrain, CharacteristicLength);
break;

case HardeningCurveType::LinearExponentialSoftening:
CalculateEquivalentStressThresholdHardeningCurveLinearExponentialSoftening(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i], CharacteristicLength,
rValues);
break;

case HardeningCurveType::CurveDefinedByPoints:
CalculateEquivalentStressThresholdHardeningCurveDefinedByPoints(
PlasticDissipation, TensileIndicatorFactor,
CompressionIndicatorFactor, eq_thresholds[i], slopes[i],
rValues, CharacteristicLength);
break;

default:
KRATOS_ERROR << " The HARDENING_CURVE of plasticity is not set or wrong..." << curve_type << std::endl;
break;
}
}

rEquivalentStressThreshold = TensileIndicatorFactor * eq_thresholds[0] + CompressionIndicatorFactor * eq_thresholds[1];
rSlope = rEquivalentStressThreshold * ((TensileIndicatorFactor * slopes[0] / eq_thresholds[0]) + (CompressionIndicatorFactor * slopes[1] / eq_thresholds[1]));
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
const Properties& r_material_properties = rValues.GetMaterialProperties();
const bool has_plastic_dissipation_limit = r_material_properties.Has(PLASTIC_DISSIPATION_LIMIT_LINEAR_SOFTENING);
const double plastic_dissipation_limit = has_plastic_dissipation_limit ? r_material_properties[PLASTIC_DISSIPATION_LIMIT_LINEAR_SOFTENING] : 0.99;
double initial_threshold;
GetInitialUniaxialThreshold(rValues, initial_threshold);

if (PlasticDissipation <= plastic_dissipation_limit){ 
rEquivalentStressThreshold = initial_threshold * std::sqrt(1.0 - PlasticDissipation);
rSlope = -0.5 * (std::pow(initial_threshold, 2.0) / (rEquivalentStressThreshold));
} else { 
rEquivalentStressThreshold =  (initial_threshold / std::sqrt(1.0 - plastic_dissipation_limit)) * (1.0 - PlasticDissipation);
rSlope = - (initial_threshold / std::sqrt(1.0 - plastic_dissipation_limit));
}
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
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double young_modulus = r_material_properties[YOUNG_MODULUS];
const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;
const double fracture_energy_compression = r_material_properties[FRACTURE_ENERGY] * std::pow(n, 2); 
const double characteristic_fracture_energy_compression = fracture_energy_compression / CharacteristicLength;

const double minimum_characteristic_fracture_energy_exponential_softening = (std::pow(yield_compression, 2)) / young_modulus;

double initial_threshold;
GetInitialUniaxialThreshold(rValues, initial_threshold);
KRATOS_ERROR_IF(characteristic_fracture_energy_compression < minimum_characteristic_fracture_energy_exponential_softening) << "The Fracture Energy is to low: " << characteristic_fracture_energy_compression << std::endl;
rEquivalentStressThreshold = initial_threshold * (1.0 - PlasticDissipation);
rSlope = - initial_threshold;
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
const Properties& r_material_properties = rValues.GetMaterialProperties();

double initial_threshold;
GetInitialUniaxialThreshold(rValues, initial_threshold);
const double ultimate_stress = r_material_properties[MAXIMUM_STRESS];              
const double max_stress_position = r_material_properties[MAXIMUM_STRESS_POSITION]; 

if (PlasticDissipation < 1.0) {
const double ro = std::sqrt(1.0 - initial_threshold / ultimate_stress);
double alpha = std::log((1.0 - (1.0 - ro) * (1.0 - ro)) / ((3.0 - ro) * (1.0 + ro) * max_stress_position));
alpha = std::exp(alpha / (1.0 - max_stress_position));
const double phi = std::pow((1.0 - ro), 2.0) + ((3.0 - ro) * (1.0 + ro) * PlasticDissipation * (std::pow(alpha, (1.0 - PlasticDissipation))));

rEquivalentStressThreshold = ultimate_stress * (2.0 * std::sqrt(phi) - phi);
rSlope = ultimate_stress * ((1.0 / std::sqrt(phi)) - 1.0) * (3.0 - ro) * (1.0 + ro) * (std::pow(alpha, (1.0 - PlasticDissipation))) *
(1.0 - std::log(alpha) * PlasticDissipation);
} else {
KRATOS_ERROR << "PlasticDissipation > 1.0 " << PlasticDissipation << std::endl;
}
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
double initial_threshold;
GetInitialUniaxialThreshold(rValues, initial_threshold);

rEquivalentStressThreshold = initial_threshold;
rSlope = 0.0;
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
const Properties& r_material_properties = rValues.GetMaterialProperties();
const Vector& curve_fitting_parameters = r_material_properties[CURVE_FITTING_PARAMETERS];

const bool has_tangency_linear_region = r_material_properties.Has(TANGENCY_REGION2);
const bool tangency_linear_region = has_tangency_linear_region ? r_material_properties[TANGENCY_REGION2] : false;

const Vector& plastic_strain_indicators = r_material_properties[PLASTIC_STRAIN_INDICATORS];
const double fracture_energy = r_material_properties[FRACTURE_ENERGY];
const double volumetric_fracture_energy = fracture_energy / CharacteristicLength;

const SizeType order_polinomial = curve_fitting_parameters.size();
const double plastic_strain_indicator_1 = plastic_strain_indicators[0];
const double plastic_strain_indicator_2 = plastic_strain_indicators[1];

double stress_indicator_1 = curve_fitting_parameters[0];
double dS_dEp = 0.0;

for (IndexType i = 1; i < order_polinomial; ++i) {
stress_indicator_1 += curve_fitting_parameters[i] * (std::pow(plastic_strain_indicator_1, i));
dS_dEp += i * curve_fitting_parameters[i] * std::pow(plastic_strain_indicator_1, i - 1);
}

double dKp_dEp = stress_indicator_1 / volumetric_fracture_energy;
if (!tangency_linear_region){
dS_dEp = 0.0; 
}

const double stress_indicator_2 =  stress_indicator_1 +  dS_dEp * (plastic_strain_indicator_2 - plastic_strain_indicator_1);

double Gt1 = 0.0;
for (IndexType i = 0; i < order_polinomial; ++i) {
Gt1 += curve_fitting_parameters[i] * (std::pow(plastic_strain_indicator_1, i + 1)) / (i + 1);
}
const double Gt2 = (stress_indicator_1 + stress_indicator_2) * (plastic_strain_indicator_2 - plastic_strain_indicator_1) * 0.5;
const double Gt3 = volumetric_fracture_energy - Gt2 - Gt1;

KRATOS_ERROR_IF(Gt3 < 0.0) << "Fracture energy too low in CurveFittingHardening of plasticity..."  << std::endl;

const double segment_threshold = (Gt2 + Gt1) / volumetric_fracture_energy;

if (PlasticDissipation <= segment_threshold) {
const double Eps = EquivalentPlasticStrain;

if (EquivalentPlasticStrain < plastic_strain_indicator_1) { 
double S_Ep = curve_fitting_parameters[0];
double dS_dEp = 0.0;
for (IndexType i = 1; i < order_polinomial; ++i) {
S_Ep += curve_fitting_parameters[i] * std::pow(Eps, i);
dS_dEp += i *  curve_fitting_parameters[i] * std::pow(Eps, i - 1);
}
dKp_dEp = S_Ep / volumetric_fracture_energy;

rEquivalentStressThreshold = S_Ep;
rSlope = dS_dEp / dKp_dEp;
} else { 
const double S_Ep = stress_indicator_1 + (stress_indicator_2 - stress_indicator_1) / (plastic_strain_indicator_2 - plastic_strain_indicator_1) * (Eps - plastic_strain_indicator_1);
double dS_dEp = (stress_indicator_2 - stress_indicator_1) / (plastic_strain_indicator_2 - plastic_strain_indicator_1);
dKp_dEp = S_Ep / volumetric_fracture_energy;

rEquivalentStressThreshold = S_Ep;
rSlope = dS_dEp / dKp_dEp;
}
} else { 
const double Eps = EquivalentPlasticStrain;
const double alpha = std::pow(stress_indicator_1, 2);
const double beta = (std::pow(stress_indicator_2, 2) - alpha) / (plastic_strain_indicator_2 - plastic_strain_indicator_1);

const double S_Ep = std::sqrt(alpha + beta * (Eps - plastic_strain_indicator_1));
const double plastic_dissipation_region_3 = PlasticDissipation - segment_threshold;

const double beta2 = 1.5 * S_Ep / Gt3;
const double alpha2 = std::sqrt((plastic_dissipation_region_3 * 2.0 * beta2 * volumetric_fracture_energy / S_Ep) + 1.0);
rEquivalentStressThreshold = S_Ep * alpha2 * (2.0 - alpha2);
rSlope = 2.0 * beta2 * volumetric_fracture_energy * (1.0 / alpha2 - 1.0);
}

}


static void CalculateEquivalentStressThresholdHardeningCurveLinearExponentialSoftening(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();
const bool has_plastic_dissipation_limit = r_material_properties.Has(PLASTIC_DISSIPATION_LIMIT_LINEAR_SOFTENING);

const double plastic_dissipation_limit = has_plastic_dissipation_limit ? r_material_properties[PLASTIC_DISSIPATION_LIMIT_LINEAR_SOFTENING] : 0.9;
const double fracture_energy = r_material_properties[FRACTURE_ENERGY];
const double volumetric_fracture_energy = fracture_energy / CharacteristicLength;
double initial_threshold;
GetInitialUniaxialThreshold(rValues, initial_threshold);

const double volumetric_fracture_energy_linear_branch = 0.5 * volumetric_fracture_energy * (plastic_dissipation_limit + 1.0);

if (PlasticDissipation <= plastic_dissipation_limit){ 
rEquivalentStressThreshold = initial_threshold * std::sqrt(1.0 - PlasticDissipation * volumetric_fracture_energy / volumetric_fracture_energy_linear_branch);
rSlope = - 0.5 * initial_threshold * (volumetric_fracture_energy / volumetric_fracture_energy_linear_branch) * std::pow(1.0 - PlasticDissipation * volumetric_fracture_energy / volumetric_fracture_energy_linear_branch, -0.5);
} else { 
const double volumetric_fracture_energy_exponential_branch = volumetric_fracture_energy * (1.0 - plastic_dissipation_limit) * std::exp((plastic_dissipation_limit + 1.0) / (std::sqrt(1.0 - std::pow(plastic_dissipation_limit, 2.0))) - 1.0);
const double initial_threshold_exponential = initial_threshold * volumetric_fracture_energy_exponential_branch / volumetric_fracture_energy * std::sqrt(1.0 - plastic_dissipation_limit * volumetric_fracture_energy / volumetric_fracture_energy_linear_branch) / (1.0 - plastic_dissipation_limit);
rEquivalentStressThreshold =  initial_threshold_exponential * (1.0 - PlasticDissipation) * volumetric_fracture_energy / volumetric_fracture_energy_exponential_branch;
rSlope = - initial_threshold_exponential * volumetric_fracture_energy / volumetric_fracture_energy_exponential_branch;
}
}


static void CalculateEquivalentStressThresholdHardeningCurveDefinedByPoints(
const double PlasticDissipation,
const double TensileIndicatorFactor,
const double CompressionIndicatorFactor,
double& rEquivalentStressThreshold,
double& rSlope,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();
const Vector& equivalent_stress_vector = r_material_properties[EQUIVALENT_STRESS_VECTOR_PLASTICITY_POINT_CURVE];
const Vector& plastic_strain_vector = r_material_properties[PLASTIC_STRAIN_VECTOR_PLASTICITY_POINT_CURVE];
const double fracture_energy = r_material_properties[FRACTURE_ENERGY];
const double volumetric_fracture_energy = fracture_energy / CharacteristicLength;
const SizeType points_hardening_curve = equivalent_stress_vector.size();

double Gt1 = 0.0;
for (IndexType i = 1; i < points_hardening_curve; ++i) {
Gt1 += 0.5 * (equivalent_stress_vector(i - 1) + equivalent_stress_vector(i)) * (plastic_strain_vector(i) - plastic_strain_vector(i - 1));
}
const double Gt2 = volumetric_fracture_energy - Gt1;

KRATOS_ERROR_IF(Gt2 < 0.0) << "Fracture energy too low in CurveDefinedByPoints of plasticity..."  << std::endl;

const double segment_threshold = (Gt1) / volumetric_fracture_energy;
if (PlasticDissipation < segment_threshold) {
IndexType i = 0;
double gf_point_region = 0.0;
double plastic_dissipation_previous_point = 0.0;
while (PlasticDissipation >= gf_point_region / volumetric_fracture_energy) {
i += 1;
plastic_dissipation_previous_point = gf_point_region / volumetric_fracture_energy;
gf_point_region += 0.5 * (equivalent_stress_vector(i - 1) + equivalent_stress_vector(i)) * (plastic_strain_vector(i) - plastic_strain_vector(i - 1));
}
const double plastic_dissipation_next_point = gf_point_region / volumetric_fracture_energy;

const double b = (std::pow(equivalent_stress_vector(i), 2.0) - std::pow(equivalent_stress_vector(i - 1), 2.0)) / (plastic_dissipation_previous_point * std::pow(equivalent_stress_vector(i), 2.0) - plastic_dissipation_next_point * std::pow(equivalent_stress_vector(i - 1), 2.0));
const double a = equivalent_stress_vector(i - 1) / std::sqrt(1.0 - b * plastic_dissipation_previous_point);
rEquivalentStressThreshold = a * std::sqrt(1.0 - b * PlasticDissipation);
rSlope = - 0.5 * std::pow(a, 2.0) * b / rEquivalentStressThreshold;

} else { 
const double b = equivalent_stress_vector(points_hardening_curve - 1) / (1.0 - segment_threshold);
const double a = b;
rEquivalentStressThreshold =  a - b * PlasticDissipation;
rSlope = - b;
}
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
double scalar_product = 0.0;
for (IndexType i = 0; i < rPlasticStrain.size(); ++i) {
scalar_product += rStressVector[i] * rPlasticStrain[i];
}


rEquivalentPlasticStrain = scalar_product / UniaxialStress;
}


static void GetInitialUniaxialThreshold(ConstitutiveLaw::Parameters& rValues, double& rThreshold)
{
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, rThreshold);
}


static void CalculateHardeningParameter(
const array_1d<double, VoigtSize>& rGFlux,
const double SlopeThreshold,
const array_1d<double, VoigtSize>& rHCapa,
double& rHardeningParameter
)
{
rHardeningParameter = SlopeThreshold;
double aux = 0.0;

for (IndexType i = 0; i < VoigtSize; ++i) {
aux += rHCapa[i] * rGFlux[i];
}
if (aux != 0.0)
rHardeningParameter *= aux;
}


static void CalculatePlasticDenominator(
const array_1d<double, VoigtSize>& rFFlux,
const array_1d<double, VoigtSize>& rGFlux,
const Matrix& rConstitutiveMatrix,
double& rHardeningParameter,
double& rPlasticDenominator
)
{
const array_1d<double, VoigtSize> delta_vector = prod(rGFlux, rConstitutiveMatrix);
double A1 = 0.0;

for (IndexType i = 0; i < VoigtSize; ++i) {
A1 += rFFlux[i] * delta_vector[i];
}
const double A2 = 0.0; 
const double A3 = rHardeningParameter;
if (std::abs(A1 + A2 + A3) > tolerance)
rPlasticDenominator = 1.0 / (A1 + A2 + A3);
else {
rPlasticDenominator = 1.0e-3 * std::numeric_limits<double>::max(); 
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
