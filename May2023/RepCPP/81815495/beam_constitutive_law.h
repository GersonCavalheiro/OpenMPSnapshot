
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) BeamConstitutiveLaw : public ConstitutiveLaw
{
public:

typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION( BeamConstitutiveLaw );




BeamConstitutiveLaw();

ConstitutiveLaw::Pointer Clone() const override;


BeamConstitutiveLaw (const BeamConstitutiveLaw& rOther);



~BeamConstitutiveLaw() override;






void GetLawFeatures(Features& rFeatures) override;


SizeType GetStrainSize() const override
{
return 1;
}


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;

protected:





private:







friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw)
}


}; 
}  
