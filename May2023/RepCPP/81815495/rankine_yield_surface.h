
#pragma once


#include "includes/checks.h"
#include "generic_yield_surface.h"
#include "custom_constitutive/auxiliary_files/plastic_potentials/rankine_plastic_potential.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"


namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class RankineYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(RankineYieldSurface);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


RankineYieldSurface()
{
}

RankineYieldSurface(RankineYieldSurface const &rOther)
{
}

RankineYieldSurface &operator=(RankineYieldSurface const &rOther)
{
return *this;
}

virtual ~RankineYieldSurface(){};



static void CalculateEquivalentStress(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
array_1d<double, Dimension> principal_stress_vector = ZeroVector(Dimension);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculatePrincipalStresses(principal_stress_vector, rPredictiveStressVector);
if constexpr (Dimension == 3)
rEquivalentStress = std::max(std::max(principal_stress_vector[0], principal_stress_vector[1]), principal_stress_vector[2]);
else 
rEquivalentStress = std::max(principal_stress_vector[0], principal_stress_vector[1]);
}


static void GetInitialUniaxialThreshold(ConstitutiveLaw::Parameters& rValues, double& rThreshold)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double yield_tension = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_TENSION];
rThreshold = std::abs(yield_tension);
}


static void CalculateDamageParameter(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

const double Gf = r_material_properties[FRACTURE_ENERGY];
const double E = r_material_properties[YOUNG_MODULUS];
const double yield_compression = r_material_properties.Has(YIELD_STRESS) ? r_material_properties[YIELD_STRESS] : r_material_properties[YIELD_STRESS_COMPRESSION];

if (r_material_properties[SOFTENING_TYPE] == static_cast<int>(SofteningType::Exponential)) {
rAParameter = 1.00 / (Gf * E / (CharacteristicLength * std::pow(yield_compression, 2)) - 0.5);
KRATOS_ERROR_IF(rAParameter < 0.0) << "Fracture energy is too low, increase FRACTURE_ENERGY..." << std::endl;
} else { 
rAParameter = -std::pow(yield_compression, 2) / (2.0 * E * Gf / CharacteristicLength);
}
}


static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rPlasticPotential,
ConstitutiveLaw::Parameters& rValues
)
{
TPlasticPotentialType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rPlasticPotential, rValues);
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues
)
{
RankinePlasticPotential<VoigtSize>::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rFFlux, rValues);
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
