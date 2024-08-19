
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class SimoJuYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(SimoJuYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


SimoJuYieldSurface()
{
}

SimoJuYieldSurface(SimoJuYieldSurface const &rOther)
{
}

SimoJuYieldSurface &operator=(SimoJuYieldSurface const &rOther)
{
return *this;
}

virtual ~SimoJuYieldSurface(){};



static void CalculateEquivalentStress(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

array_1d<double, Dimension> principal_stress_vector;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculatePrincipalStresses(principal_stress_vector, rPredictiveStressVector);

const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = std::abs(yield_compression / yield_tension);

double SumA = 0.0, SumB = 0.0, SumC = 0.0, ere0, ere1;
for (std::size_t cont = 0; cont < 2; ++cont) {
SumA += std::abs(principal_stress_vector[cont]);
SumB += 0.5 * (principal_stress_vector[cont] + std::abs(principal_stress_vector[cont]));
SumC += 0.5 * (-principal_stress_vector[cont] + std::abs(principal_stress_vector[cont]));
}
ere0 = SumB / SumA;
ere1 = SumC / SumA;

double auxf = 0.0;
for (std::size_t cont = 0; cont < VoigtSize; ++cont) {
auxf += rStrainVector[cont] * rPredictiveStressVector[cont]; 
}
rEquivalentStress = std::sqrt(auxf);
rEquivalentStress *= (ere0 * n + ere1);
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double yield_compression = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
rThreshold = std::abs(yield_compression / std::sqrt(r_material_properties[YOUNG_MODULUS]));
}


static void CalculateDamageParameter(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double fracture_energy = r_material_properties[FRACTURE_ENERGY];
const bool has_symmetric_yield_stress = r_material_properties.Has(YIELD_STRESS);
const double yield_compression = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];
const double yield_tension = has_symmetric_yield_stress ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
const double n = yield_compression / yield_tension;

if (r_material_properties[SOFTENING_TYPE] == static_cast<int>(SofteningType::Exponential)) {
rAParameter = 1.0 / (fracture_energy * n * n / (CharacteristicLength * std::pow(yield_compression, 2)) - 0.5);
KRATOS_ERROR_IF(rAParameter < 0.0) << "Fracture energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(yield_compression, 2) / (2.0 * fracture_energy * n * n / CharacteristicLength);
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
const array_1d<double, VoigtSize>& Deviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues
)
{
KRATOS_ERROR << "Yield surface derivative not defined for SimoJu..." << std::endl;
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
return false;
}


static double GetScaleFactorTension(const Properties& rMaterialProperties)
{
const double yield_compression = rMaterialProperties.Has(YIELD_STRESS) ? rMaterialProperties[YIELD_STRESS] : rMaterialProperties[YIELD_STRESS_COMPRESSION];
const double yield_tension = rMaterialProperties.Has(YIELD_STRESS) ? rMaterialProperties[YIELD_STRESS] : rMaterialProperties[YIELD_STRESS_TENSION];
return std::sqrt(rMaterialProperties[YOUNG_MODULUS]) * yield_tension / yield_compression;
}





protected:







private:








}; 





} 
