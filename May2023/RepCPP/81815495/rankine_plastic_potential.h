
#pragma once


#include "generic_plastic_potential.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class RankinePlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(RankinePlasticPotential);


RankinePlasticPotential()
{
}

RankinePlasticPotential(RankinePlasticPotential const &rOther)
{
}

RankinePlasticPotential &operator=(RankinePlasticPotential const &rOther)
{
return *this;
}

virtual ~RankinePlasticPotential(){};




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

double J3, lode_angle;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(rDeviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

double c1, c3, c2;
double checker = std::abs(lode_angle * 180.0 / Globals::Pi);
const double sqrt_3 = std::sqrt(3.0);

if (std::abs(checker) < 29.0) { 
const double sqrt_J2 = std::sqrt(J2);
const double square_sin_3_lode = std::pow(std::sin(3.0 * lode_angle), 2);
const double angle = lode_angle + Globals::Pi / 6.0;
const double dLode_dJ2 = (3.0 * sqrt_3 * J3) / (4.0 * J2 * J2 * sqrt_J2 * std::sqrt(1.0 - square_sin_3_lode));
const double dLode_dJ3 = -sqrt_3 / (2.0 * J2 * sqrt_J2 * std::sqrt(1.0 - square_sin_3_lode));
c1 = 1.0 / 3.0;
c2 = 2.0 * sqrt_3 / 3.0 * (std::cos(angle) / (2.0 * sqrt_J2) - 2.0 * sqrt_3 * sqrt_J2 / 3.0 * std::sin(angle) * dLode_dJ2) * 2.0 * sqrt_J2;
c3 = -2.0 * std::sqrt(3.0 * J2) / 3.0 * std::sin(angle) * dLode_dJ3;
} else { 
const double friction_angle = rValues.GetMaterialProperties()[FRICTION_ANGLE] * Globals::Pi / 180.0;
const double sin_phi = std::sin(friction_angle);
const double CFL = -sqrt_3 * (3.0 - sin_phi) / (3.0 * sin_phi - 3.0);
c1 = CFL * 2.0 * sin_phi / (sqrt_3 * (3.0 - sin_phi));
c2 = CFL;
c3 = 0.0;
}
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
