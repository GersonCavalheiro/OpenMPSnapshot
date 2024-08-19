
#pragma once


#include "generic_plastic_potential.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class VonMisesPlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(VonMisesPlasticPotential);


VonMisesPlasticPotential()
{
}

VonMisesPlasticPotential(VonMisesPlasticPotential const &rOther)
{
}

VonMisesPlasticPotential &operator=(VonMisesPlasticPotential const &rOther)
{
return *this;
}

virtual ~VonMisesPlasticPotential(){};




static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
ConstitutiveLaw::Parameters& rValues
)
{
array_1d<double, VoigtSize> first_vector, second_vector, third_vector;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateThirdVector(rDeviator, J2, third_vector);

const double c1 = 0.0;
const double c2 = std::sqrt(3.0);
const double c3 = 0.0;

noalias(rGFlux) = c1 * first_vector + c2 * second_vector + c3 * third_vector;
}


static int Check(const Properties& rMaterialProperties)
{
return 0;
}






protected:








private:








}; 





} 
