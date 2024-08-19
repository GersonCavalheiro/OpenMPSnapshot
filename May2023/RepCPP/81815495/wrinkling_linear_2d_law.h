
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) WrinklingLinear2DLaw
: public ConstitutiveLaw
{
public:


typedef ProcessInfo ProcessInfoType;

typedef ConstitutiveLaw    BaseType;

typedef std::size_t        SizeType;

KRATOS_CLASS_POINTER_DEFINITION( WrinklingLinear2DLaw );

enum class WrinklingType {
Taut,
Slack,
Wrinkle
};



WrinklingLinear2DLaw();


WrinklingLinear2DLaw (const WrinklingLinear2DLaw& rOther);


~WrinklingLinear2DLaw() override = default;




ConstitutiveLaw::Pointer Clone() const override;


ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;


SizeType WorkingSpaceDimension() override;


SizeType GetStrainSize() const override;



bool Has(const Variable<bool>& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<int>& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<double>& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<Vector>& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<Matrix>& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<array_1d<double, 3 > >& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool Has(const Variable<array_1d<double, 6 > >& rThisVariable) override
{
return mpConstitutiveLaw->Has(rThisVariable);
}


bool& GetValue(const Variable<bool>& rThisVariable,bool& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


int& GetValue(const Variable<int>& rThisVariable,int& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


double& GetValue(const Variable<double>& rThisVariable,double& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


Vector& GetValue(const Variable<Vector>& rThisVariable,Vector& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


Matrix& GetValue(const Variable<Matrix>& rThisVariable,Matrix& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


array_1d<double, 3 >& GetValue(const Variable<array_1d<double, 3 >>& rThisVariable,array_1d<double, 3 >& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}



array_1d<double, 6 >& GetValue(const Variable<array_1d<double, 6 >>& rThisVariable,array_1d<double, 6 >& rValue) override
{
mpConstitutiveLaw->GetValue(rThisVariable,rValue);
return rValue;
}


bool& CalculateValue(Parameters& rParameterValues,const Variable<bool>& rThisVariable,
bool& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


int& CalculateValue(Parameters& rParameterValues,
const Variable<int>& rThisVariable,int& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


double& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,double& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


Vector& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,Vector& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


Matrix& CalculateValue(ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,Matrix& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


array_1d<double, 3 >& CalculateValue(Parameters& rParameterValues,
const Variable<array_1d<double, 3 >>& rThisVariable,array_1d<double, 3 >& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


array_1d<double, 6 >& CalculateValue(Parameters& rParameterValues,
const Variable<array_1d<double, 6 >>& rThisVariable,array_1d<double, 6 >& rValue) override
{
mpConstitutiveLaw->CalculateValue(rParameterValues,rThisVariable,rValue);
return rValue;
}


void SetValue(const Variable<bool>& rThisVariable,const bool& rValue,
const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<int>& rThisVariable,const int& rValue,
const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<double>& rThisVariable,const double& rValue,
const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<Vector>& rThisVariable,const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<Matrix>& rThisVariable,const Matrix& rValue,
const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<array_1d<double, 3 >>& rThisVariable,
const array_1d<double, 3 >& rValue,const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void SetValue(const Variable<array_1d<double, 6 >>& rThisVariable,
const array_1d<double, 6 >& rValue,const ProcessInfo& rCurrentProcessInfo) override
{
mpConstitutiveLaw->SetValue(rThisVariable, rValue, rCurrentProcessInfo);
}


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;



void CalculateMaterialResponsePK2 (Parameters& rValues) override;


void InitializeMaterialResponsePK2 (Parameters& rValues) override;



void FinalizeMaterialResponsePK2 (Parameters& rValues) override;



void ResetMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void GetLawFeatures(Features& rFeatures) override;



int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;


void PrincipalVector(Vector& rPrincipalVector, const Vector& rNonPrincipalVector);



void CheckWrinklingState(WrinklingType& rWrinklingState, const Vector& rStress, const Vector& rStrain,
Vector& rWrinklingDirectionVector);

protected:






private:



ConstitutiveLaw::Pointer mpConstitutiveLaw;





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw )
rSerializer.save("ConstitutiveLaw", mpConstitutiveLaw);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw)
rSerializer.load("ConstitutiveLaw", mpConstitutiveLaw);
}


}; 
}  
