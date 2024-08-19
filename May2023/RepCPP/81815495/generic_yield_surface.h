
#pragma once


#include "includes/define.h"
#include "includes/checks.h"
#include "includes/serializer.h"
#include "includes/properties.h"
#include "includes/constitutive_law.h"
#include "utilities/math_utils.h"
#include "custom_utilities/constitutive_law_utilities.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TPlasticPotentialType>
class GenericYieldSurface
{
public:

typedef TPlasticPotentialType PlasticPotentialType;

static constexpr SizeType Dimension = PlasticPotentialType::Dimension;

static constexpr SizeType VoigtSize = PlasticPotentialType::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(GenericYieldSurface);



GenericYieldSurface()
{
}

GenericYieldSurface(GenericYieldSurface const &rOther)
{
}

GenericYieldSurface &operator=(GenericYieldSurface const &rOther)
{
return *this;
}

virtual ~GenericYieldSurface(){};




static void CalculateEquivalentStress(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const Vector& rStrainVector,
double& rEquivalentStress,
ConstitutiveLaw::Parameters& rValues
)
{
}


static void GetInitialUniaxialThreshold(
ConstitutiveLaw::Parameters& rValues,
double& rThreshold
)
{
}


static void CalculateDamageParameter(
ConstitutiveLaw::Parameters& rValues,
double& rAParameter,
const double CharacteristicLength
)
{
}


static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double& J2,
array_1d<double, VoigtSize>& rDerivativePlasticPotential,
ConstitutiveLaw::Parameters& rValues
)
{
TPlasticPotentialType::CalculatePlasticPotentialDerivative(rPredictiveStressVector, rDeviator, J2, rDerivativePlasticPotential, rValues);
}


static void CalculateYieldSurfaceDerivative(
const array_1d<double, VoigtSize>& StressVector,
const array_1d<double, VoigtSize>& Deviator,
const double J2,
array_1d<double, VoigtSize>& rFFlux,
ConstitutiveLaw::Parameters& rValues)
{
}


static int Check(const Properties& rMaterialProperties)
{
return TPlasticPotentialType::Check(rMaterialProperties);
}






protected:







private:








}; 





} 
