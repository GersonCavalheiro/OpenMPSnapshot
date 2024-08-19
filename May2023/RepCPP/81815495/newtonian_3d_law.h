
#pragma once



#include "fluid_constitutive_law.h"

namespace Kratos
{


class KRATOS_API(FLUID_DYNAMICS_APPLICATION) Newtonian3DLaw : public FluidConstitutiveLaw
{
public:

typedef ConstitutiveLaw         BaseType;
typedef std::size_t             SizeType;


KRATOS_CLASS_POINTER_DEFINITION(Newtonian3DLaw);




Newtonian3DLaw();


ConstitutiveLaw::Pointer Clone() const override;


Newtonian3DLaw (const Newtonian3DLaw& rOther);



~Newtonian3DLaw() override;




SizeType WorkingSpaceDimension() override;


SizeType GetStrainSize() const override;


void CalculateMaterialResponseCauchy (Parameters& rValues) override;


void CalculateDerivative(
Parameters& rParameterValues,
const Variable<double>& rFunctionVariable,
const Variable<double>& rDerivativeVariable,
double& rOutput) override;


void CalculateDerivative(
Parameters& rParameterValues,
const Variable<Vector>& rFunctionVariable,
const Variable<double>& rDerivativeVariable,
Vector& rOutput) override;


void CalculateDerivative(
Parameters& rParameterValues,
const Variable<Matrix>& rFunctionVariable,
const Variable<double>& rDerivativeVariable,
Matrix& rOutput) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo) const override;




std::string Info() const override;

protected:


double GetEffectiveViscosity(ConstitutiveLaw::Parameters& rParameters) const override;

private:







friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;

}; 
}  
