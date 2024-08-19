
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class MohrCoulombYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(MohrCoulombYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


MohrCoulombYieldSurface()
{
}

MohrCoulombYieldSurface(MohrCoulombYieldSurface const &rOther)
{
}

MohrCoulombYieldSurface &operator=(MohrCoulombYieldSurface const &rOther)
{
return *this;
}

virtual ~MohrCoulombYieldSurface(){};




static void CalculateEquivalentStress(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
double I1, J2, J3, lode_angle;
array_1d<double, VoigtSize> deviator = ZeroVector(VoigtSize);

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateI1Invariant(rPredictiveStressVector, I1);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rPredictiveStressVector, I1, deviator, J2);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(deviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

const Properties& r_material_properties = rValues.GetMaterialProperties();
const double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0;

rEquivalentStress = (std::cos(lode_angle) - std::sin(lode_angle) * std::sin(friction_angle) / std::sqrt(3.0)) * std::sqrt(J2) +
I1 * std::sin(friction_angle) / 3.0;
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();
const double cohesion = r_material_properties[COHESION];
const double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0;

rThreshold = cohesion * std::cos(friction_angle);
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
double equivalent_yield;
GetInitialUniaxialThreshold(rValues, equivalent_yield);
if (r_material_properties[SOFTENING_TYPE] == static_cast<int>(SofteningType::Exponential)) {
rAParameter = 1.00 / (fracture_energy * young_modulus / (CharacteristicLength * std::pow(equivalent_yield, 2)) - 0.5);
KRATOS_ERROR_IF(rAParameter < 0.0) << "Fracture Energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(equivalent_yield, 2) / (2.0 * young_modulus * fracture_energy / CharacteristicLength);
}
}


static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rDerivativePlasticPotential,
ConstitutiveLaw::Parameters& rValues
)
{
TPlasticPotentialType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rDerivativePlasticPotential, rValues);
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues
)
{
array_1d<double, VoigtSize> first_vector, second_vector, third_vector;
const Properties& r_material_properties = rValues.GetMaterialProperties();
const double friction_angle = r_material_properties[FRICTION_ANGLE] * Globals::Pi / 180.0;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateThirdVector(rDeviator, J2, third_vector);

double J3, lode_angle;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(rDeviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

double c1, c3, c2;
double checker = std::abs(lode_angle * 180.0 / Globals::Pi);

if (std::abs(checker) < 29.0) { 
c1 = std::sin(friction_angle) / 3.0;
c3 = (std::sqrt(3.0) * std::sin(lode_angle) + std::sin(friction_angle) * std::cos(lode_angle)) /
(2.0 * J2 * std::cos(3.0 * lode_angle));
c2 = 0.5 * std::cos(lode_angle)*(1.0 + std::tan(lode_angle) * std::sin(3.0 * lode_angle) +
std::sin(friction_angle) * (std::tan(3.0 * lode_angle) - std::tan(lode_angle)) / std::sqrt(3.0));
} else { 
c1 = 3.0 * (2.0 * std::sin(friction_angle) / (std::sqrt(3.0) * (3.0 - std::sin(friction_angle))));
c2 = 1.0;
c3 = 0.0;
}

noalias(rFFlux) = c1 * first_vector + c2 * second_vector + c3 * third_vector;
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(COHESION)) << "COHESION is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRICTION_ANGLE)) << "FRICTION_ANGLE is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRACTURE_ENERGY)) << "FRACTURE_ENERGY is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YOUNG_MODULUS)) << "YOUNG_MODULUS is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YIELD_STRESS)) << "YIELD_STRESS is not a defined value" << std::endl;

return TPlasticPotentialType::Check(rMaterialProperties);
}


static bool IsWorkingWithTensionThreshold()
{
return true;
}


static double GetScaleFactorTension(const Properties& rMaterialProperties)
{
return 1.0;
}






protected:







private:








}; 





} 
