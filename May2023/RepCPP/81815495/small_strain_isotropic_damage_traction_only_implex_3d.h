
#pragma once



#include "small_strain_isotropic_damage_implex_3d.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainIsotropicDamageTractionOnlyImplex3D
: public SmallStrainIsotropicDamageImplex3D
{
public:

typedef ProcessInfo ProcessInfoType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainIsotropicDamageTractionOnlyImplex3D);



SmallStrainIsotropicDamageTractionOnlyImplex3D();


SmallStrainIsotropicDamageTractionOnlyImplex3D(const SmallStrainIsotropicDamageTractionOnlyImplex3D& rOther);


~SmallStrainIsotropicDamageTractionOnlyImplex3D() override;


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
