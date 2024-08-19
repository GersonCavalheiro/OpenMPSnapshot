
#pragma once

#include "custom_constitutive/dam_joint_3D_law.hpp"

namespace Kratos
{

class KRATOS_API(DAM_APPLICATION) DamJoint2DLaw : public DamJoint3DLaw
{

public:

KRATOS_CLASS_POINTER_DEFINITION(DamJoint2DLaw);


DamJoint2DLaw()
{
}

ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<DamJoint2DLaw>(DamJoint2DLaw(*this));
}

DamJoint2DLaw (const DamJoint2DLaw& rOther) : DamJoint3DLaw(rOther)
{
}

~DamJoint2DLaw() override
{
}


void GetLawFeatures(Features& rFeatures) override;


protected:



void ComputeEquivalentStrain(ConstitutiveLawVariables& rVariables,
Parameters& rValues) override;

void ComputeConstitutiveMatrix(Matrix& rConstitutiveMatrix,
ConstitutiveLawVariables& rVariables,
Parameters& rValues) override;

void ComputeStressVector(Vector& rStressVector,
ConstitutiveLawVariables& rVariables,
Parameters& rValues) override;


private:


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw )
}

}; 
}  
