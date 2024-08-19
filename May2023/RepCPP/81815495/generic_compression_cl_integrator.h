
#pragma once


#include "includes/define.h"
#include "includes/checks.h"
#include "includes/serializer.h"
#include "includes/properties.h"
#include "utilities/math_utils.h"
#include "constitutive_laws_application_variables.h"

namespace Kratos
{



typedef std::size_t SizeType;





template <class TYieldSurfaceType>
class GenericCompressionConstitutiveLawIntegratorDplusDminusDamage
{
public:


typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension = YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef typename YieldSurfaceType::PlasticPotentialType PlasticPotentialType;

KRATOS_CLASS_POINTER_DEFINITION(GenericCompressionConstitutiveLawIntegratorDplusDminusDamage);

GenericCompressionConstitutiveLawIntegratorDplusDminusDamage()
{
}

GenericCompressionConstitutiveLawIntegratorDplusDminusDamage(GenericCompressionConstitutiveLawIntegratorDplusDminusDamage const &rOther)
{
}

GenericCompressionConstitutiveLawIntegratorDplusDminusDamage &operator=(GenericCompressionConstitutiveLawIntegratorDplusDminusDamage const &rOther)
{
return *this;
}

virtual ~GenericCompressionConstitutiveLawIntegratorDplusDminusDamage()
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
const int softening_type = r_material_properties.Has(SOFTENING_TYPE_COMPRESSION) ? r_material_properties[SOFTENING_TYPE_COMPRESSION] : r_material_properties[SOFTENING_TYPE];
double damage_parameter;
CalculateDamageParameterCompression(rValues, damage_parameter, CharacteristicLength);

switch (softening_type)
{
case static_cast<int>(SofteningType::Linear):
CalculateLinearDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
case static_cast<int>(SofteningType::Exponential):
CalculateExponentialDamage(UniaxialStress, rThreshold, damage_parameter, CharacteristicLength, rValues, rDamage);
break;
default:
KRATOS_ERROR << "SOFTENING_TYPE not defined or wrong..." << softening_type << std::endl;
break;
}
rPredictiveStressVector *= (1.0 - rDamage);
}


static void CalculateDamageParameterCompression(
ConstitutiveLaw::Parameters& rValues,
double& rDamageParameter,
const double CharacteristicLength
)
{
const double fracture_energy_compression = rValues.GetMaterialProperties()[FRACTURE_ENERGY_COMPRESSION];
ConstitutiveLaw::Parameters modified_values = rValues;
auto r_properties = modified_values.GetMaterialProperties();
r_properties.SetValue(FRACTURE_ENERGY, fracture_energy_compression);
modified_values.SetMaterialProperties(r_properties);
TYieldSurfaceType::CalculateDamageParameter(modified_values, rDamageParameter, CharacteristicLength);
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
if (YieldSurfaceType::IsWorkingWithTensionThreshold()) {
ConstitutiveLaw::Parameters modified_ones = rValues;
const double yield_compression = modified_ones.GetMaterialProperties()[YIELD_STRESS_COMPRESSION];
Properties material_props = modified_ones.GetMaterialProperties();
material_props.SetValue(YIELD_STRESS_TENSION, yield_compression);
modified_ones.SetMaterialProperties(material_props);
TYieldSurfaceType::GetInitialUniaxialThreshold(modified_ones, rThreshold);  
} else {
TYieldSurfaceType::GetInitialUniaxialThreshold(rValues, rThreshold); 
}   
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
GetInitialUniaxialThreshold(rValues, initial_threshold);
rDamage = 1.0 - (initial_threshold / UniaxialStress) * std::exp(DamageParameter *
(1.0 - UniaxialStress / initial_threshold));
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
GetInitialUniaxialThreshold(rValues, initial_threshold);
rDamage = (1.0 - initial_threshold / UniaxialStress) / (1.0 + DamageParameter);
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(SOFTENING_TYPE)) << "MAXIMUM_STRESS is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YIELD_STRESS_TENSION)) << "YIELD_STRESS_TENSION is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YIELD_STRESS_COMPRESSION)) << "YIELD_STRESS_COMPRESSION is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(YOUNG_MODULUS)) << "YOUNG_MODULUS is not a defined value" << std::endl;
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(FRACTURE_ENERGY)) << "FRACTURE_ENERGY is not a defined value" << std::endl;

return TYieldSurfaceType::Check(rMaterialProperties);
}






protected:








private:








};
} 
