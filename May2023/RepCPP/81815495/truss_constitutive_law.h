
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TrussConstitutiveLaw : public ConstitutiveLaw
{
public:

typedef ProcessInfo      ProcessInfoType;
typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION( TrussConstitutiveLaw );




TrussConstitutiveLaw();

ConstitutiveLaw::Pointer Clone() const override;


TrussConstitutiveLaw (const TrussConstitutiveLaw& rOther);



~TrussConstitutiveLaw() override;






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

array_1d<double, 3 > & GetValue(const Variable<array_1d<double, 3 > >& rThisVariable,
array_1d<double, 3 > & rValue) override;

double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,double& rValue) override;

Vector& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue) override;



array_1d<double, 3 > & CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<array_1d<double, 3 > >& rVariable,
array_1d<double, 3 > & rValue) override;

void CalculateMaterialResponsePK2(Parameters& rValues) override;


void FinalizeMaterialResponsePK2(Parameters& rValues) override
{
};



double CalculateStressElastic(ConstitutiveLaw::Parameters& rParameterValues);

protected:





private:






friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw);
}


}; 
}  
