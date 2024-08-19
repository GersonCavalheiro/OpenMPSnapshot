
#pragma once


#include "includes/checks.h"
#include "generic_plastic_potential.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class DruckerPragerPlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(DruckerPragerPlasticPotential);


DruckerPragerPlasticPotential()
{
}

DruckerPragerPlasticPotential(DruckerPragerPlasticPotential const &rOther)
{
}

DruckerPragerPlasticPotential &operator=(DruckerPragerPlasticPotential const &rOther)
{
return *this;
}

virtual ~DruckerPragerPlasticPotential(){};




static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
ConstitutiveLaw::Parameters& rValues
)
{
const Properties& r_material_properties = rValues.GetMaterialProperties();

array_1d<double, VoigtSize> first_vector, second_vector, third_vector;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);

const double dilatancy = r_material_properties[DILATANCY_ANGLE] * Globals::Pi / 180.0;
const double sin_dil = std::sin(dilatancy);
const double Root3 = std::sqrt(3.0);

const double CFL = -Root3 * (3.0 - sin_dil) / (3.0 * sin_dil - 3.0);
const double c1 = CFL * 2.0 * sin_dil / (Root3 * (3.0 - sin_dil));
const double c2 = CFL;

noalias(rGFlux) = c1 * first_vector + c2 * second_vector;
}


static int Check(const Properties& rMaterialProperties)
{
KRATOS_ERROR_IF_NOT(rMaterialProperties.Has(DILATANCY_ANGLE)) << "DILATANCY_ANGLE is not a defined value" << std::endl;

return 0;
}






protected:








private:








}; 





} 
