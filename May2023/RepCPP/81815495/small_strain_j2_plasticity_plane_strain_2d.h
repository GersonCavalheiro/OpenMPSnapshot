
#pragma once



#include "small_strain_j2_plasticity_3d.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SmallStrainJ2PlasticityPlaneStrain2D
: public SmallStrainJ2Plasticity3D
{
public:


typedef ProcessInfo ProcessInfoType;
typedef SmallStrainJ2Plasticity3D BaseType;
typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(SmallStrainJ2PlasticityPlaneStrain2D);



SmallStrainJ2PlasticityPlaneStrain2D();


SmallStrainJ2PlasticityPlaneStrain2D(const SmallStrainJ2PlasticityPlaneStrain2D& rOther);


~SmallStrainJ2PlasticityPlaneStrain2D() override;


ConstitutiveLaw::Pointer Clone() const override;




void GetLawFeatures(Features& rFeatures) override;


SizeType WorkingSpaceDimension() override
{
return 2;
};


SizeType GetStrainSize() const override
{
return 4;
};


void PrintData(std::ostream& rOStream) const override {
rOStream << "Linear J2 Plasticity Plane Strain 2D constitutive law\n";
};

protected:






void CalculateStressResponse(
ConstitutiveLaw::Parameters& rValues,
Vector& rPlasticStrain,
double& rAccumulatedPlasticStrain
) override;


void CalculateTangentMatrix(const double DeltaGamma, const double NormStressTrial,
const Vector &YieldFunctionNormalVector,
const Properties &rMaterialProperties,
const double AccumulatedPlasticStrain, Matrix &rElasticityTensor) override;


void CalculateElasticMatrix(const Properties &rMaterialProperties, Matrix &rElasticityTensor) override;




private:







friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
} 
