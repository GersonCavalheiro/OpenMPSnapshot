
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"

namespace Kratos
{


typedef std::size_t SizeType;




template<class TPlasticPotentialType>
class DruckerPragerYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(DruckerPragerYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


DruckerPragerYieldSurface()
{
}

DruckerPragerYieldSurface(DruckerPragerYieldSurface const &rOther)
{
}

DruckerPragerYieldSurface &operator=(DruckerPragerYieldSurface const &rOther)
{
return *this;
}

virtual ~DruckerPragerYieldSurface(){};



static void CalculateEquivalentStress(
array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0; 
const double sin_phi = std::sin(friction_angle);
const double root_3 = std::sqrt(3.0);

if (friction_angle < tolerance) {
friction_angle = 32.0 * Globals::Pi / 180.0;
KRATOS_WARNING("DruckerPragerYieldSurface") << "Friction Angle not defined, assumed equal to 32 " << std::endl;
}

double I1, J2;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateI1Invariant(rPredictiveStressVector, I1);
array_1d<double, VoigtSize> deviator;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rPredictiveStressVector, I1, deviator, J2);


const double CFL = -root_3 * (3.0 - sin_phi) / (3.0 * sin_phi - 3.0);
const double TEN0 = 2.0 * I1 * sin_phi / (root_3 * (3.0 - sin_phi)) + std::sqrt(J2);
rEquivalentStress = (CFL * TEN0);
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double yield_tension = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0; 
const double sin_phi = std::sin(friction_angle);
rThreshold = std::abs(yield_tension * (3.0 + sin_phi) / (3.0 * sin_phi - 3.0));
}


static void CalculateDamageParameter(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double Gf = r_material_properties[FRACTURE_ENERGY];
const double E = r_material_properties[YOUNG_MODULUS];
const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;

if (r_material_properties[SOFTENING_TYPE] == static_cast<int>(SofteningType::Exponential)) {
rAParameter = 1.00 / (Gf * n * n * E / (CharacteristicLength * std::pow(yield_compression, 2)) - 0.5);
KRATOS_ERROR_IF(rAParameter < 0.0) << "Fracture energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(yield_compression, 2) / (2.0 * E * Gf * n * n / CharacteristicLength);
}
}


static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
ConstitutiveLaw::Parameters& rValues
)
{
TPlasticPotentialType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rGFlux, rValues);
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

array_1d<double, VoigtSize> first_vector, second_vector, third_vector;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);

const double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0;;
const double sin_phi = std::sin(friction_angle);
const double Root3 = std::sqrt(3.0);

const double CFL = -Root3 * (3.0 - sin_phi) / (3.0 * sin_phi - 3.0);
const double c1 = CFL * 2.0 * sin_phi / (Root3 * (3.0 - sin_phi));
const double c2 = CFL;

noalias(rFFlux) = c1 * first_vector + c2 * second_vector;
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
return true;
}


static double GetScaleFactorTension(const Properties& rMaterialProperties)
{
const double friction_angle = rMaterialProperties[FRICTION_ANGLE] * Globals::Pi / 180.0; 
const double sin_phi = std::sin(friction_angle);
return 1.0 / std::abs((3.0 + sin_phi) / (3.0 * sin_phi - 3.0));
}






protected:







private:








}; 





} 