
#pragma once



#include "small_strain_isotropic_damage_3d.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainIsotropicDamageTractionOnly3D
: public SmallStrainIsotropicDamage3D

{
public:

typedef ProcessInfo ProcessInfoType;
typedef ConstitutiveLaw BaseType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainIsotropicDamageTractionOnly3D);



SmallStrainIsotropicDamageTractionOnly3D();


SmallStrainIsotropicDamageTractionOnly3D(const SmallStrainIsotropicDamageTractionOnly3D& rOther);


~SmallStrainIsotropicDamageTractionOnly3D() override;


ConstitutiveLaw::Pointer Clone() const override;



void ComputePositiveStressVector(
Vector& rStressVectorPos,
Vector& rStressVector) override;

private:











friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
} 
