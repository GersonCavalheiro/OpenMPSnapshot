
#pragma once

#include <cmath>

#include "includes/serializer.h"
#include "includes/checks.h"
#include "includes/constitutive_law.h"

#include "dam_application_variables.h"
#include "poromechanics_application_variables.h"

namespace Kratos
{

class KRATOS_API(DAM_APPLICATION) DamJoint3DLaw : public ConstitutiveLaw
{

public:

KRATOS_CLASS_POINTER_DEFINITION(DamJoint3DLaw);


DamJoint3DLaw()
{
}

ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<DamJoint3DLaw>(DamJoint3DLaw(*this));
}

DamJoint3DLaw (const DamJoint3DLaw& rOther) : ConstitutiveLaw(rOther)
{
}

~DamJoint3DLaw() override {}


void GetLawFeatures(Features& rFeatures) override;

int Check(const Properties& rMaterialProperties, const GeometryType& rElementGeometry, const ProcessInfo& rCurrentProcessInfo) const override;

void InitializeMaterial( const Properties& rMaterialProperties,const GeometryType& rElementGeometry,const Vector& rShapeFunctionsValues ) override;


void CalculateMaterialResponseCauchy (Parameters & rValues) override;

void FinalizeMaterialResponseCauchy (Parameters & rValues) override;


double& GetValue( const Variable<double>& rThisVariable, double& rValue ) override;

void SetValue( const Variable<double>& rVariable, const double& rValue, const ProcessInfo& rCurrentProcessInfo ) override;


protected:

struct ConstitutiveLawVariables
{
double YoungModulus;
double YieldStress;

Matrix CompressionMatrix;

double EquivalentStrain;
bool LoadingFlag;
double LoadingFunction;
};


double mStateVariable;


virtual void InitializeConstitutiveLawVariables(ConstitutiveLawVariables& rVariables,
Parameters& rValues);

virtual void ComputeEquivalentStrain(ConstitutiveLawVariables& rVariables,
Parameters& rValues);

virtual void CheckLoadingFunction(ConstitutiveLawVariables& rVariables,
Parameters& rValues);

virtual void ComputeConstitutiveMatrix(Matrix& rConstitutiveMatrix,
ConstitutiveLawVariables& rVariables,
Parameters& rValues);

virtual void ComputeStressVector(Vector& rStressVector,
ConstitutiveLawVariables& rVariables,
Parameters& rValues);


private:


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw )
}

}; 
}  
