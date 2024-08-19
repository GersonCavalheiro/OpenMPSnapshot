
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainIsotropicPlasticityFactory
: public ConstitutiveLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainIsotropicPlasticityFactory);



SmallStrainIsotropicPlasticityFactory()
{
}


~SmallStrainIsotropicPlasticityFactory() override
{
}


ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;








protected:







private:








}; 

} 