
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{


class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) LinearElasticOrthotropic2DLaw : public ConstitutiveLaw
{
public:

typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION(LinearElasticOrthotropic2DLaw);




LinearElasticOrthotropic2DLaw();


ConstitutiveLaw::Pointer Clone() const override;


LinearElasticOrthotropic2DLaw(const LinearElasticOrthotropic2DLaw& rOther);


~LinearElasticOrthotropic2DLaw() override;





SizeType GetStrainSize() const override
{
return 3;
};


void GetLawFeatures(Features& rFeatures) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters & rValues) override;


bool& GetValue(const Variable<bool>& rThisVariable, bool& rValue) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo) const override;

protected:


void CalculateGreenLagrangeStrain(const Matrix & rRightCauchyGreen,
Vector& rStrainVector);


virtual void CalculateStress(const Vector &rStrainVector,
const Matrix &rConstitutiveMatrix,
Vector& rStressVector);



virtual void CalculateLinearElasticMatrix(Matrix& rConstitutiveMatrix,
const Properties& rMaterialProperties);


bool CheckParameters(ConstitutiveLaw::Parameters& rValues);

private:



friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
}

}; 
}  