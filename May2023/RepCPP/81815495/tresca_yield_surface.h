
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class TrescaYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(TrescaYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


TrescaYieldSurface()
{
}

TrescaYieldSurface(TrescaYieldSurface const &rOther)
{
}

TrescaYieldSurface &operator=(TrescaYieldSurface const &rOther)
{
return *this;
}

virtual ~TrescaYieldSurface(){};




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

rEquivalentStress = 2.0 * std::cos(lode_angle) * std::sqrt(J2);
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double yield_tension = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
rThreshold = std::abs(yield_tension);
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
KRATOS_ERROR_IF(rAParameter < 0.0) << "Fracture energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(yield_compression, 2) / (2.0 * young_modulus * fracture_energy * n * n / CharacteristicLength);
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
array_1d<double, VoigtSize> second_vector, third_vector;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateThirdVector(rDeviator, J2, third_vector);

double J3, lode_angle;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(rDeviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

const double checker = std::abs(lode_angle * 180.0 / Globals::Pi);

double c2, c3;
if (std::abs(checker) < 29.0) { 
c2 = 2.0 * (std::cos(lode_angle) + std::sin(lode_angle) * std::tan(3.0 * lode_angle));
c3 = std::sqrt(3.0) * std::sin(lode_angle) / (J2 * std::cos(3.0 * lode_angle));
} else {
c2 = std::sqrt(3.0);
c3 = 0.0;
}

noalias(rFFlux) = c2 * second_vector + c3 * third_vector;
}


static int Check(const Properties& rMaterialProperties)
{
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
return 1.0;
}






protected:







private:








}; 





} 
