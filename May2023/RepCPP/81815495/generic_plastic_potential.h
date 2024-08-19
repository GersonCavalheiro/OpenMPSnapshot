
#pragma once


#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/properties.h"
#include "utilities/math_utils.h"
#include "constitutive_laws_application_variables.h"
#include "custom_utilities/advanced_constitutive_law_utilities.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class GenericPlasticPotential
{
public:

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(GenericPlasticPotential);


GenericPlasticPotential()
{
}

GenericPlasticPotential(GenericPlasticPotential const &rOther)
{
}

GenericPlasticPotential &operator=(GenericPlasticPotential const &rOther)
{
return *this;
}

virtual ~GenericPlasticPotential(){};




static void CalculatePlasticPotentialDerivative(
const array_1d<double, VoigtSize>& rPredictiveStressVector,
const array_1d<double, VoigtSize>& rDeviator,
const double J2,
array_1d<double, VoigtSize>& rGFlux,
ConstitutiveLaw::Parameters& rValues
)
{
}


static int Check(const Properties& rMaterialProperties)
{
return 0;
}






protected:








private:








}; 





} 
