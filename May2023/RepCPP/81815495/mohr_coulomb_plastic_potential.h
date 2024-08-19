
#pragma once


#include "generic_plastic_potential.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class MohrCoulombPlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(MohrCoulombPlasticPotential);

static constexpr double tolerance = std::numeric_limits<double>::epsilon();


MohrCoulombPlasticPotential()
{
}

MohrCoulombPlasticPotential(MohrCoulombPlasticPotential const &rOther)
{
}

MohrCoulombPlasticPotential &operator=(MohrCoulombPlasticPotential const &rOther)
{
return *this;
}

virtual ~MohrCoulombPlasticPotential(){};




static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
ConstitutiveLaw::Parameters& rValues
)
{
array_1d<double, VoigtSize> first_vector, second_vector, third_vector;
const Properties& r_material_properties = rValues.GetMaterialProperties();
const double dilatancy = r_material_properties[DILATANCY_ANGLE] * Globals::Pi / 180.0;

AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateFirstVector(first_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateSecondVector(rDeviator, J2, second_vector);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateThirdVector(rDeviator, J2, third_vector);

double J3, lode_angle;
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateJ3Invariant(rDeviator, J3);
AdvancedConstitutiveLawUtilities<VoigtSize>::CalculateLodeAngle(J2, J3, lode_angle);

double c1, c3, c2;
double checker = std::abs(lode_angle * 180.0 / Globals::Pi);

if (std::abs(checker) < 29.0) {
c1 = std::sin(dilatancy);
c3 = (std::sqrt(3.0) * std::sin(lode_angle) + std::sin(dilatancy) * std::cos(lode_angle)) /
(2.0 * J2 * std::cos(3.0 * lode_angle));
c2 = 0.5 * std::cos(lode_angle)*(1.0 + std::tan(lode_angle) * std::sin(3.0 * lode_angle) +
std::sin(dilatancy) * (std::tan(3.0 * lode_angle) - std::tan(lode_angle)) / std::sqrt(3.0));
} else { 
c1 = 3.0 * (2.0 * std::sin(dilatancy) / (std::sqrt(3.0) * (3.0 - std::sin(dilatancy))));
c2 = 1.0;
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
