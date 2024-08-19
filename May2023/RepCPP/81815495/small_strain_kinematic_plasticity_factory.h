
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainKinematicPlasticityFactory
: public ConstitutiveLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainKinematicPlasticityFactory);



SmallStrainKinematicPlasticityFactory()
{
}


~SmallStrainKinematicPlasticityFactory() override
{
}


ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;








protected:







private:








}; 

} 
