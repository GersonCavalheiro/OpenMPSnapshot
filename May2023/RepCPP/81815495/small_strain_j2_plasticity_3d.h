
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainJ2Plasticity3D
: public ConstitutiveLaw
{
public:


typedef ProcessInfo ProcessInfoType;
typedef ConstitutiveLaw BaseType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainJ2Plasticity3D);



SmallStrainJ2Plasticity3D();


SmallStrainJ2Plasticity3D(const SmallStrainJ2Plasticity3D& rOther);


~SmallStrainJ2Plasticity3D() override;


ConstitutiveLaw::Pointer Clone() const override;




void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return 3;
};


SizeType GetStrainSize() const override
{
return 6;
};


bool Has(const Variable<double>& rThisVariable) override;


void SetValue(
const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue
) override;


void InitializeMaterial(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff(Parameters& rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


void PrintData(std::ostream& rOStream) const override {
rOStream << "Linear J2 Plasticity 3D constitutive law\n";
};

protected:



Vector mPlasticStrain; 
double mAccumulatedPlasticStrain; 




using ConstitutiveLaw::CalculateStressResponse;
virtual void CalculateStressResponse(ConstitutiveLaw::Parameters& rValues,
Vector& rPlasticStrain,
double& rAccumulatedPlasticStrain );


double YieldFunction(
const double NormDeviationStress,
const Properties& rMaterialProperties,
const double AccumulatedPlasticStrain
);


double GetAccumPlasticStrainRate(const double NormStressTrial, const Properties &rMaterialProperties,
const double AccumulatedPlasticStrainOld);


double GetSaturationHardening(const Properties& rMaterialProperties, const double);


double GetPlasticPotential(const Properties& rMaterialProperties,
const double accumulated_plastic_strain);


virtual void CalculateTangentMatrix(const double DeltaGamma, const double NormStressTrial,
const Vector &rYFNormalVector,
const Properties &rMaterialProperties,
const double AccumulatedPlasticStrain, Matrix &rTMatrix);


virtual void CalculateElasticMatrix(const Properties &rMaterialProperties, Matrix &rElasticMatrix);






private:







friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
} 
