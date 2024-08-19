
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainViscoplasticity3D
: public ConstitutiveLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainViscoplasticity3D);



GenericSmallStrainViscoplasticity3D()
{
}


GenericSmallStrainViscoplasticity3D(
ConstitutiveLaw::Pointer pPlasticityLaw,
ConstitutiveLaw::Pointer pViscousLaw) : mpPlasticityConstitutiveLaw(pPlasticityLaw), mpViscousConstitutiveLaw(pViscousLaw)
{
}


ConstitutiveLaw::Pointer Clone() const override
{
auto p_law = Kratos::make_shared<GenericSmallStrainViscoplasticity3D>(*this);

p_law->SetPlasticityConstitutiveLaw(mpPlasticityConstitutiveLaw->Clone());
p_law->SetViscousConstitutiveLaw(mpViscousConstitutiveLaw->Clone());

return p_law;
}

GenericSmallStrainViscoplasticity3D(GenericSmallStrainViscoplasticity3D const &rOther)
: ConstitutiveLaw(rOther), mpPlasticityConstitutiveLaw(rOther.mpPlasticityConstitutiveLaw), mpViscousConstitutiveLaw(rOther.mpViscousConstitutiveLaw)
{
}


~GenericSmallStrainViscoplasticity3D() override
{
}




ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;


SizeType WorkingSpaceDimension() override
{
return 3;
};


SizeType GetStrainSize() const override
{
return 6;
};


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeSolutionStep(
const Properties &rMaterialProperties,
const GeometryType &rElementGeometry,
const Vector &rShapeFunctionsValues,
const ProcessInfo &rCurrentProcessInfo) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;

Vector &GetValue(const Variable<Vector> &rThisVariable, Vector &rValue) override;

double &GetValue(const Variable<double> &rThisVariable, double &rValue) override;

bool Has(const Variable<double> &rThisVariable) override;


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


bool RequiresInitializeMaterialResponse() override
{
return false;
}

double &CalculateValue(
Parameters &rParameterValues,
const Variable<double> &rThisVariable,
double &rValue) override;






protected:



ConstitutiveLaw::Pointer GetPlasticityConstitutiveLaw()
{
return mpPlasticityConstitutiveLaw;
}

void SetPlasticityConstitutiveLaw(ConstitutiveLaw::Pointer pPlasticityConstitutiveLaw)
{
mpPlasticityConstitutiveLaw = pPlasticityConstitutiveLaw;
}

ConstitutiveLaw::Pointer GetViscousConstitutiveLaw()
{
return mpViscousConstitutiveLaw;
}

void SetViscousConstitutiveLaw(ConstitutiveLaw::Pointer pViscousConstitutiveLaw)
{
mpViscousConstitutiveLaw = pViscousConstitutiveLaw;
}





private:


ConstitutiveLaw::Pointer mpPlasticityConstitutiveLaw;
ConstitutiveLaw::Pointer mpViscousConstitutiveLaw;




void CalculateElasticMatrix(
Matrix &rElasticityTensor,
const Properties &rMaterialProperties);





friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("PlasticityConstitutiveLaw", mpPlasticityConstitutiveLaw);
rSerializer.save("ViscousConstitutiveLaw", mpViscousConstitutiveLaw);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("PlasticityConstitutiveLaw", mpPlasticityConstitutiveLaw);
rSerializer.load("ViscousConstitutiveLaw", mpViscousConstitutiveLaw);
}


}; 

} 
