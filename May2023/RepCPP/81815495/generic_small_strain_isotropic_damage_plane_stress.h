
#pragma once



#include "generic_small_strain_isotropic_damage.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainIsotropicDamagePlaneStress
: public  GenericSmallStrainIsotropicDamage<TConstLawIntegratorType>
{
public:

typedef GenericSmallStrainIsotropicDamage<TConstLawIntegratorType> BaseType;

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainIsotropicDamagePlaneStress);

GenericSmallStrainIsotropicDamagePlaneStress()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericSmallStrainIsotropicDamagePlaneStress<TConstLawIntegratorType>>(*this);
}


GenericSmallStrainIsotropicDamagePlaneStress(const GenericSmallStrainIsotropicDamagePlaneStress &rOther)
: BaseType(rOther)
{
}


~GenericSmallStrainIsotropicDamagePlaneStress() override
{
}


Matrix& CalculateValue(
ConstitutiveLaw::Parameters &rParameterValues,
const Variable<Matrix> &rThisVariable,
Matrix &rValue) override;







protected:







private:








friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
}


};

} 
