
#pragma once

#include "custom_constitutive/truss_constitutive_law.h"

namespace Kratos
{




class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) MultiLinearElastic1DLaw : public TrussConstitutiveLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MultiLinearElastic1DLaw );


MultiLinearElastic1DLaw();

ConstitutiveLaw::Pointer Clone() const override;


MultiLinearElastic1DLaw (const MultiLinearElastic1DLaw& rOther);



~MultiLinearElastic1DLaw() override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;

double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,double& rValue) override;

private:

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, TrussConstitutiveLaw);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, TrussConstitutiveLaw);
}


}; 

}  
