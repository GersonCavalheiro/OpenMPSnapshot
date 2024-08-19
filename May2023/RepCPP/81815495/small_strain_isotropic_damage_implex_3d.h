
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainIsotropicDamageImplex3D
: public ElasticIsotropic3D
{
public:

typedef ProcessInfo ProcessInfoType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainIsotropicDamageImplex3D);



SmallStrainIsotropicDamageImplex3D();


SmallStrainIsotropicDamageImplex3D(const SmallStrainIsotropicDamageImplex3D& rOther);


~SmallStrainIsotropicDamageImplex3D() override;


ConstitutiveLaw::Pointer Clone() const override;




void GetLawFeatures(Features& rFeatures) override;


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


Vector& GetValue(
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


void SetValue(
const Variable<Vector>& rThisVariable,
const Vector& rValue,
const ProcessInfo& rProcessInfo
) override;


void InitializeMaterial(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;


void CalculateStressResponse(ConstitutiveLaw::Parameters& rValues,
Vector& rInternalVariables) override;


void CalculateMaterialResponsePK2(Parameters& rValues) override;


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


void FinalizeMaterialResponseCauchy(Parameters& rValues) override;


double& CalculateValue(Parameters& rValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& CalculateValue(Parameters& rValues,
const Variable<Vector>& rThisVariable,
Vector& rValue) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


void PrintData(std::ostream& rOStream) const override {
rOStream << "Small Strain Isotropic Damage 3D constitutive law with Implex intergration scheme\n";
};

protected:


double mStrainVariable;
double mStrainVariablePrevious;




virtual void ComputePositiveStressVector(
Vector& rStressVectorPos,
Vector& rStressVector);


double EvaluateHardeningModulus(
double StrainVariable,
const Properties &rMaterialProperties);


double EvaluateHardeningLaw(
double StrainVariable,
const Properties &rMaterialProperties);


private:











friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
} 
