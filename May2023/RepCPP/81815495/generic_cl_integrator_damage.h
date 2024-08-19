
#pragma once


#include "includes/define.h"
#include "includes/checks.h"
#include "includes/serializer.h"
#include "includes/properties.h"
#include "utilities/math_utils.h"
#include "constitutive_laws_application_variables.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"

namespace Kratos
{


typedef std::size_t SizeType;





template <class TYieldSurfaceType>
class GenericConstitutiveLawIntegratorDamage
{
public:

typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension = YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef typename YieldSurfaceType::PlasticPotentialType PlasticPotentialType;

KRATOS_CLASS_POINTER_DEFINITION(GenericConstitutiveLawIntegratorDamage);

GenericConstitutiveLawIntegratorDamage()
{
}

GenericConstitutiveLawIntegratorDamage(GenericConstitutiveLawIntegratorDamage const &rOther)
{
}

GenericConstitutiveLawIntegratorDamage &operator=(GenericConstitutiveLawIntegratorDamage const &rOther)
{
return *this;
}

virtual ~GenericConstitutiveLawIntegratorDamage()
{
}




static void IntegrateStressVector(
array_1d<double, VoigtSize>& rPredictiveStressVector,
const double UniaxialStress,
double& rDamage,
double& rThreshold,
ConstitutiveLaw::Parameters& rValues,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const int softening_type = r_material_properties[SOFTENING_TYPE];
double damage_parameter;
TYieldSurfaceType::CalculateDamageParameter(rValues, damage_parameter, CharacteristicLength);

switch (softening_type)
{
case static_cast<int>(SofteningType::Linear):
CalculateLinearDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
case static_cast<int>(SofteningType::Exponential):
CalculateExponentialDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
case static_cast<int>(SofteningType::HardeningDamage):
CalculateHardeningDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
case static_cast<int>(SofteningType::CurveFittingDamage):
CalculateCurveFittingDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
default:
KRATOS_ERROR << "SOFTENING_TYPE not defined or wrong..." << softening_type << std::endl;
break;
}
rDamage = (rDamage > 0.99999) ? 0.99999 : rDamage;
rDamage = (rDamage < 0.0) ? 0.0 : rDamage;
rPredictiveStressVector *= (1.0 - rDamage);
}


static void CalculateExponentialDamage(
const double UniaxialStress,
const double Threshold,
const double DamageParameter,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues,
double& rDamage
)
{
double initial_threshold;
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, initial_threshold);
rDamage = 1.0 - (initial_threshold / UniaxialStress) * std::exp(DamageParameter *
(1.0 - UniaxialStress / initial_threshold));
}


static void CalculateHardeningDamage(
const double UniaxialStress,
const double Threshold,
const double DamageParameter,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues,
double& rDamage
)
{
const auto &r_mat_props = rValues.GetMaterialProperties();
const double max_stress = r_mat_props[MAXIMUM_STRESS];
const double Gf = r_mat_props[FRACTURE_ENERGY];
const double E = r_mat_props[YOUNG_MODULUS];
const bool has_symmetric_yield_stress = r_mat_props.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_mat_props[YIELD_STRESS] : r_mat_props[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_mat_props[YIELD_STRESS] : r_mat_props[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;

double initial_threshold;
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, initial_threshold);

const double re = max_stress / initial_threshold;
const double rp = 1.5 * re;
const double Ad = (rp - re) / re;
const double Ad_tilda = Ad * (std::pow(rp, 3) - 3.0 * rp + 2.0 / 3.0) / (6.0 * re * std::pow((rp - 1.0), 2));
const double Hd = 1.0 / (2.0 * (E * Gf * n * n / max_stress / max_stress / CharacteristicLength - 0.5 * rp / re - Ad_tilda));

const double r = UniaxialStress / initial_threshold;

if (r <= rp) {
rDamage = Ad * re / r * std::pow(((r - 1.0) / (rp - 1.0)), 2);
} else {
rDamage = 1.0 - re / r + Hd * (1.0 - rp / r);
}
}


static void CalculateLinearDamage(
const double UniaxialStress,
const double Threshold,
const double DamageParameter,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues,
double& rDamage
)
{
double initial_threshold;
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, initial_threshold);
rDamage = (1.0 - initial_threshold / UniaxialStress) / (1.0 + DamageParameter);
}


static void CalculateCurveFittingDamage(
const double UniaxialStress,
const double Threshold,
const double DamageParameter,
const double CharacteristicLength,
ConstitutiveLaw::Parameters& rValues,
double& rDamage
)
{
const Properties &r_mat_props = rValues.GetMaterialProperties();
const double fracture_energy = r_mat_props[FRACTURE_ENERGY];
const double volumetric_fracture_energy = fracture_energy / CharacteristicLength;
const double yield_stress = r_mat_props[YIELD_STRESS];
const double E = r_mat_props[YOUNG_MODULUS];
const Vector& strain_damage_curve = r_mat_props[STRAIN_DAMAGE_CURVE]; 
const Vector& stress_damage_curve = r_mat_props[STRESS_DAMAGE_CURVE]; 
const SizeType curve_points = strain_damage_curve.size() - 1;

double volumentric_fracture_energy_first_region = 0.5 * std::pow(yield_stress, 2.0) / E; 
for (IndexType i = 1; i <= curve_points; ++i) {
volumentric_fracture_energy_first_region += 0.5 * (stress_damage_curve[i-1] + stress_damage_curve[i])
* (strain_damage_curve[i] - strain_damage_curve[i-1]);
const double irreversibility_damage_check = (stress_damage_curve[i] - stress_damage_curve[i-1]) / (strain_damage_curve[i] - strain_damage_curve[i-1]);
KRATOS_ERROR_IF(irreversibility_damage_check > E)<< "The defined S-E curve induces negative damage at region " << i << std::endl;
}
KRATOS_ERROR_IF(volumentric_fracture_energy_first_region > volumetric_fracture_energy) << "The Fracture Energy is too low: " << fracture_energy << std::endl;

const double predictive_stress_end_first_region = strain_damage_curve[curve_points] * E;
if (UniaxialStress < predictive_stress_end_first_region){ 
for (IndexType i = 1; i <= curve_points; ++i) {
if (UniaxialStress < strain_damage_curve[i] * E){
const double current_integrated_stress = stress_damage_curve[i-1] + (UniaxialStress / E - strain_damage_curve[i-1])
* (stress_damage_curve[i] - stress_damage_curve[i-1]) / (strain_damage_curve[i] - strain_damage_curve[i-1]);
rDamage = 1.0 - current_integrated_stress / UniaxialStress;
break;
}
}
} else { 
const double volumentric_fracture_energy_second_region = volumetric_fracture_energy - volumentric_fracture_energy_first_region;
rDamage = 1.0 - stress_damage_curve[curve_points] / UniaxialStress * std::exp(stress_damage_curve[curve_points] * (strain_damage_curve[curve_points] * E - UniaxialStress) / (E * volumentric_fracture_energy_second_region));
}
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, rThreshold);
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(SOFTENING_TYPE)) << "SOFTENING_TYPE is not a defined value" << std::endl;
return TYieldSurfaceType::Check(rMaterialProperties);
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& rStressVector,
array_1d<double, VoigtSize>& rFlux,
ConstitutiveLaw::Parameters& rValues
)
{
array_1d<double, VoigtSize> deviator = ZeroVector(6);
double J2;
const double I1 = rStressVector[0] + rStressVector[1] + rStressVector[2];
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rStressVector, I1, deviator, J2);
YieldSurfaceType::CalculateYieldSurfaceDerivative(rStressVector, deviator, J2, rFlux, rValues);
}






protected:








private:








}; 





} 
