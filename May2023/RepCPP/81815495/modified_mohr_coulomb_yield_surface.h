
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class ModifiedMohrCoulombYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(ModifiedMohrCoulombYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


ModifiedMohrCoulombYieldSurface()
{
}

ModifiedMohrCoulombYieldSurface(ModifiedMohrCoulombYieldSurface const &rOther)
{
}

ModifiedMohrCoulombYieldSurface &operator=(ModifiedMohrCoulombYieldSurface const &rOther)
{
return *this;
}

virtual ~ModifiedMohrCoulombYieldSurface(){};



static void CalculateEquivalentStress(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0; 

if (friction_angle < tolerance) {
friction_angle = 32.0 * Globals::Pi / 180.0;
KRATOS_WARNING("ModifiedMohrCoulombYieldSurface") << "Friction Angle not defined, assumed equal to 32 deg " << std::endl;
}

double theta;
const double R = std::abs(yield_compression / yield_tension);
const double Rmorh = std::pow(std::tan((Globals::Pi / 4.0) + friction_angle / 2.0), 2);
const double alpha_r = R / Rmorh;
const double sin_phi = std::sin(friction_angle);

double I1, J2, J3;
array_1d<double, VoigtSize> deviator = ZeroVector(VoigtSize);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateI1Invariant(rPredictiveStressVector, I1);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rPredictiveStressVector, I1, deviator, J2);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(deviator, J3);

const double K1 = 0.5 * (1.0 + alpha_r) - 0.5 * (1.0 - alpha_r) * sin_phi;
const double K2 = 0.5 * (1.0 + alpha_r) - 0.5 * (1.0 - alpha_r) / sin_phi;
const double K3 = 0.5 * (1.0 + alpha_r) * sin_phi - 0.5 * (1.0 - alpha_r);

if (std::abs(I1) < tolerance) {
rEquivalentStress = 0.0;
} else {
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, theta);
rEquivalentStress = (2.0 * std::tan(Globals::Pi * 0.25 + friction_angle * 0.5) / std::cos(friction_angle)) * ((I1 * K3 / 3.0) +
std::sqrt(J2) * (K1 * std::cos(theta) - K2 * std::sin(theta) * sin_phi / std::sqrt(3.0)));
}
}


static void GetInitialUniaxialThreshold(ConstitutiveLaw::Parameters& rValues, double& rThreshold)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double yield_compression = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
rThreshold = std::abs(yield_compression);
}


static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& GFlux,
ConstitutiveLaw::Parameters& rValues
)
{
TPlasticPotentialType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, GFlux, rValues);
}


static void CalculateDamageParameter(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double fracture_energy = r_material_properties[FRACTURE_ENERGY];
const double young_modulus = r_material_properties[YOUNG_MODULUS];
const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;

if (r_material_properties[SOFTENING_TYPE] == static_cast<int>(SofteningType::Exponential)) {
rAParameter = 1.00 / (fracture_energy * n * n * young_modulus / (CharacteristicLength * std::pow(yield_compression, 2)) - 0.5);
KRATOS_DEBUG_ERROR_IF(rAParameter < 0.0) << "Fracture energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(yield_compression, 2) / (2.0 * young_modulus * fracture_energy * n * n / CharacteristicLength);
}
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

array_1d<double, VoigtSize> first_vector, second_vector, third_vector;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateThirdVector(rDeviator, J2, third_vector);

double J3, lode_angle;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(rDeviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

const double checker = std::abs(lode_angle * 180.0 / Globals::Pi);

double c1, c2, c3;
double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0;

if (friction_angle < tolerance) {
friction_angle = 32.0 * Globals::Pi / 180.0;
KRATOS_WARNING("ModifiedMohrCoulombYieldSurface") << "Friction Angle not defined, assumed equal to 32 deg " << std::endl;
}

const double sin_phi = std::sin(friction_angle);
const double cons_phi = std::cos(friction_angle);
const double sin_theta = std::sin(lode_angle);
const double cos_theta = std::cos(lode_angle);
const double cos_3theta = std::cos(3.0 * lode_angle);
const double tan_theta = std::tan(lode_angle);
const double tan_3theta = std::tan(3.0 * lode_angle);
const double Root3 = std::sqrt(3.0);

const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double compr_yield = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double tens_yield = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = compr_yield / tens_yield;

const double angle_phi = (Globals::Pi * 0.25) + friction_angle * 0.5;
const double alpha = n / (std::tan(angle_phi) * std::tan(angle_phi));

const double CFL = 2.0 * std::tan(angle_phi) / cons_phi;

const double K1 = 0.5 * (1.0 + alpha) - 0.5 * (1.0 - alpha) * sin_phi;
const double K2 = 0.5 * (1.0 + alpha) - 0.5 * (1.0 - alpha) / sin_phi;
const double K3 = 0.5 * (1.0 + alpha) * sin_phi - 0.5 * (1.0 - alpha);

if (std::abs(sin_phi) > tolerance)
c1 = CFL * K3 / 3.0;
else
c1 = 0.0; 

if (std::abs(checker) < 29.0) { 
c2 = cos_theta * CFL * (K1 * (1.0 + tan_theta * tan_3theta) + K2 * sin_phi * (tan_3theta - tan_theta) / Root3);
c3 = CFL * (K1 * Root3 * sin_theta + K2 * sin_phi * cos_theta) / (2.0 * J2 * cos_3theta);
} else {
c3 = 0.0;
double aux = 1.0;
if (lode_angle > tolerance)
aux = -1.0;
c2 = 0.5 * CFL * (K1 * Root3 + aux * K2 * sin_phi / Root3);
}
noalias(rFFlux) = c1 * first_vector + c2 * second_vector + c3 * third_vector;
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRICTION_ANGLE)) << "FRICTION_ANGLE is not a defined value" << std::endl;
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
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRACTURE_ENERGY)) << "FRACTURE_ENERGY is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YOUNG_MODULUS)) << "YOUNG_MODULUS is not a defined value" << std::endl;

return TPlasticPotentialType::Check(rMaterialProperties);
}


static bool IsWorkingWithTensionThreshold()
{
return false;
}


static double GetScaleFactorTension(const Properties& rMaterialProperties)
{
const double yield_compression = rMaterialProperties[YIELD_STRESS_COMPRESSION];
const double yield_tension = rMaterialProperties[YIELD_STRESS_TENSION];
return yield_compression / yield_tension;
}







protected:







private:








}; 





} 
