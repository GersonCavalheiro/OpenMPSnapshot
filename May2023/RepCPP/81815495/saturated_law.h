
#pragma once

#include <string>
#include <iostream>
#include "includes/define.h"


#include "includes/serializer.h"
#include "custom_retention/retention_law.h"

#include "geo_mechanics_application_variables.h"

namespace Kratos
{

class KRATOS_API(GEO_MECHANICS_APPLICATION) SaturatedLaw
: public RetentionLaw
{
public:

using BaseType = RetentionLaw;

using GeometryType = Geometry<Node>;

using SizeType = std::size_t;

KRATOS_CLASS_POINTER_DEFINITION( SaturatedLaw );

SaturatedLaw();

RetentionLaw::Pointer Clone() const override;

SaturatedLaw(const SaturatedLaw& rOther);

~SaturatedLaw() override;

void InitializeMaterial(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;

void Initialize(Parameters &rParameters) override;

void InitializeSolutionStep(Parameters &rParameters) override;

double CalculateSaturation(Parameters &rParameters) override;

double CalculateEffectiveSaturation(Parameters &rParameters) override;

double CalculateDerivativeOfSaturation(Parameters &rParameters) override;

double CalculateRelativePermeability(Parameters &rParameters) override;

double CalculateBishopCoefficient(Parameters &rParameters) override;

void Finalize(Parameters &rParameters) override;

void FinalizeSolutionStep(Parameters &rParameters) override;


double& CalculateValue(RetentionLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;



int Check(const Properties& rMaterialProperties,
const ProcessInfo& rCurrentProcessInfo) override;

private:
friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, RetentionLaw )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, RetentionLaw)
}

}; 
}  
