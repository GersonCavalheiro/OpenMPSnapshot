
#pragma once



#include "includes/checks.h"
#include "includes/constitutive_law.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) PlasticityIsotropicKinematicJ2
: public ConstitutiveLaw
{
public:


typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;

static constexpr SizeType VoigtSize = 6;   

static constexpr SizeType Dimension = VoigtSize == 6 ? 3 : 2;

typedef array_1d<double, VoigtSize> BoundedArrayType;

typedef Matrix MatrixType;

KRATOS_CLASS_POINTER_DEFINITION(PlasticityIsotropicKinematicJ2);



PlasticityIsotropicKinematicJ2();


PlasticityIsotropicKinematicJ2(const PlasticityIsotropicKinematicJ2& rOther);


~PlasticityIsotropicKinematicJ2() override;


ConstitutiveLaw::Pointer Clone() const override;




void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return Dimension;
};


SizeType GetStrainSize() const override
{
return VoigtSize;
};


bool Has(const Variable<double>& rThisVariable) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue
) override;


void SetValue(
const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void InitializeMaterial(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;


void FinalizeSolutionStep(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo) override;


void InitializeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


void PrintData(std::ostream& rOStream) const override {
rOStream << "Linear J2 Plasticity 3D constitutive law\n";
};

protected:



BoundedArrayType mPlasticStrain;           
double           mEquivalentPlasticStrain; 




double YieldFunction(
const double      NormDeviatoricStress,
const Properties& rMaterialProperties
);


void CalculateResponse6(
ConstitutiveLaw::Parameters& rValues,
BoundedArrayType&            rPlasticStrain,
double&                      rAccumulatedPlasticStrain
);


void CalculateTangentTensor6(
const double            DeltaGamma,
const double            NormStressTrial,
const BoundedArrayType& rYieldFunctionNormalVector,
const Properties&       rMaterialProperties,
MatrixType&             rTangentTensor
);


void CalculateElasticMatrix6(
const Properties& rMaterialProperties,
MatrixType&       rElasticityTensor
);






private:







friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
} 
