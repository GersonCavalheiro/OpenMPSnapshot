
#pragma once



#include "custom_constitutive/linear_plane_strain.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainIsotropicDamage
: public std::conditional<TConstLawIntegratorType::VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type
{
public:

static constexpr SizeType Dimension = TConstLawIntegratorType::Dimension;

static constexpr SizeType VoigtSize = TConstLawIntegratorType::VoigtSize;

typedef typename std::conditional<VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type BaseType;

typedef array_1d<double, VoigtSize> BoundedArrayType;

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainIsotropicDamage);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();
static constexpr double threshold_tolerance = 1.0e-5;



GenericSmallStrainIsotropicDamage()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericSmallStrainIsotropicDamage<TConstLawIntegratorType>>(*this);
}


GenericSmallStrainIsotropicDamage(const GenericSmallStrainIsotropicDamage &rOther)
: BaseType(rOther),
mDamage(rOther.mDamage),
mThreshold(rOther.mThreshold)
{
}


~GenericSmallStrainIsotropicDamage() override
{
}




void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;

void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;


bool Has(const Variable<double> &rThisVariable) override;


bool Has(const Variable<Vector> &rThisVariable) override;


bool Has(const Variable<Matrix> &rThisVariable) override;


void SetValue(
const Variable<double> &rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<Vector> &rThisVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


double& GetValue(
const Variable<double> &rThisVariable,
double& rValue
) override;


Vector& GetValue(
const Variable<Vector> &rThisVariable,
Vector& rValue
) override;


Matrix& GetValue(
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


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



void CalculateTangentTensor(ConstitutiveLaw::Parameters &rValues);






protected:




double& GetThreshold() { return mThreshold; }
double& GetDamage() { return mDamage; }

void SetThreshold(const double toThreshold) { mThreshold = toThreshold; }
void SetDamage(const double toDamage) { mDamage = toDamage; }




private:


double mDamage = 0.0;
double mThreshold = 0.0;







friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("Damage", mDamage);
rSerializer.save("Threshold", mThreshold);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("Damage", mDamage);
rSerializer.load("Threshold", mThreshold);
}


}; 

} 
