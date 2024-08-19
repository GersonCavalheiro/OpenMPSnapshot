
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
class GenericTensionConstitutiveLawIntegratorDplusDminusDamage
{
public:


typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension = YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef typename YieldSurfaceType::PlasticPotentialType PlasticPotentialType;

KRATOS_CLASS_POINTER_DEFINITION(GenericTensionConstitutiveLawIntegratorDplusDminusDamage);

GenericTensionConstitutiveLawIntegratorDplusDminusDamage()
{
}

GenericTensionConstitutiveLawIntegratorDplusDminusDamage(GenericTensionConstitutiveLawIntegratorDplusDminusDamage const &rOther)
{
}

GenericTensionConstitutiveLawIntegratorDplusDminusDamage &operator=(GenericTensionConstitutiveLawIntegratorDplusDminusDamage const &rOther)
{
return *this;
}

virtual ~GenericTensionConstitutiveLawIntegratorDplusDminusDamage()
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
default:
KRATOS_ERROR << "SOFTENING_TYPE not defined or wrong..." << softening_type << std::endl;
break;
}
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






protected:








private:








};
} 
