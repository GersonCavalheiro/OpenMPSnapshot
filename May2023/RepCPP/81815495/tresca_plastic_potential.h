
#pragma once


#include "generic_plastic_potential.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class TrescaPlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(TrescaPlasticPotential);


TrescaPlasticPotential()
{
}

TrescaPlasticPotential(TrescaPlasticPotential const &rOther)
{
}

TrescaPlasticPotential &operator=(TrescaPlasticPotential const &rOther)
{
return *this;
}

virtual ~TrescaPlasticPotential(){};




static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
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
if (checker < 29.0) {
c2 = 2.0 * (std::cos(lode_angle) + std::sin(lode_angle) * std::tan(3.0 * lode_angle));
c3 = std::sqrt(3.0) * std::sin(lode_angle) / (J2 * std::cos(3.0 * lode_angle));
} else {
c2 = std::sqrt(3.0);
c3 = 0.0;
}

noalias(rGFlux) = c2 * second_vector + c3 * third_vector;
}


static int Check(const Properties& rMaterialProperties)
{
return 0;
}






protected:








private:








}; 





} 
