
#pragma once



#include "small_strain_isotropic_damage_3d.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainIsotropicDamagePlaneStrain2D
: public SmallStrainIsotropicDamage3D

{
public:


typedef ProcessInfo ProcessInfoType;
typedef SmallStrainIsotropicDamage3D BaseType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainIsotropicDamagePlaneStrain2D);


SmallStrainIsotropicDamagePlaneStrain2D();


SmallStrainIsotropicDamagePlaneStrain2D(const SmallStrainIsotropicDamagePlaneStrain2D& rOther);


~SmallStrainIsotropicDamagePlaneStrain2D() override;


ConstitutiveLaw::Pointer Clone() const override;


void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return 2;
};


SizeType GetStrainSize() const override
{
return 3;
};

void PrintData(std::ostream& rOStream) const override {
rOStream << "Linear Isotropic Damage Plane Strain 2D constitutive law\n";
};

protected:


void CalculateElasticMatrix(Matrix &rElasticMatrix, Parameters &rMaterialProperties) override;



private:






friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 
} 
