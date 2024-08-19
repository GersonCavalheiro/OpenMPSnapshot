
#pragma once



#include "custom_constitutive/small_strains/plasticity/generic_small_strain_isotropic_plasticity.h"

namespace Kratos
{


typedef std::size_t SizeType;





template<class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericFiniteStrainIsotropicPlasticity
: public GenericSmallStrainIsotropicPlasticity<TConstLawIntegratorType>
{
public:

static constexpr SizeType Dimension = TConstLawIntegratorType::Dimension;

static constexpr SizeType VoigtSize = TConstLawIntegratorType::VoigtSize;

typedef GenericSmallStrainIsotropicPlasticity<TConstLawIntegratorType> BaseType;

typedef array_1d<double, VoigtSize> BoundedArrayType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(GenericFiniteStrainIsotropicPlasticity);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;



GenericFiniteStrainIsotropicPlasticity()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericFiniteStrainIsotropicPlasticity<TConstLawIntegratorType>>(*this);
}


GenericFiniteStrainIsotropicPlasticity(const GenericFiniteStrainIsotropicPlasticity &rOther)
: BaseType(rOther)
{
}


~GenericFiniteStrainIsotropicPlasticity() override
{
}




void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;

void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


bool RequiresInitializeMaterialResponse() override
{
return false;
}


double& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


Matrix& CalculateValue(
ConstitutiveLaw::Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;






protected:







private:





void CalculateTangentTensor(
ConstitutiveLaw::Parameters &rValues,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy
);






}; 

} 