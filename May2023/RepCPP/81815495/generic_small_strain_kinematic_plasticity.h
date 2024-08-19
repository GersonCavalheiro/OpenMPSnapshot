
#pragma once



#include "custom_constitutive/elastic_isotropic_3d.h"
#include "custom_constitutive/linear_plane_strain.h"

namespace Kratos
{


typedef std::size_t SizeType;





template <class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericSmallStrainKinematicPlasticity
: public std::conditional<TConstLawIntegratorType::VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type
{
public:

static constexpr SizeType Dimension = TConstLawIntegratorType::Dimension;

static constexpr SizeType VoigtSize = TConstLawIntegratorType::VoigtSize;

typedef typename std::conditional<VoigtSize == 6, ElasticIsotropic3D, LinearPlaneStrain >::type BaseType;

typedef array_1d<double, VoigtSize> BoundedArrayType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(GenericSmallStrainKinematicPlasticity);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;



GenericSmallStrainKinematicPlasticity()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericSmallStrainKinematicPlasticity<TConstLawIntegratorType>>(*this);
}


GenericSmallStrainKinematicPlasticity(const GenericSmallStrainKinematicPlasticity &rOther)
: BaseType(rOther),
mPlasticDissipation(rOther.mPlasticDissipation),
mThreshold(rOther.mThreshold),
mPlasticStrain(rOther.mPlasticStrain),
mPreviousStressVector(rOther.mPreviousStressVector),
mBackStressVector(rOther.mBackStressVector)
{
}


~GenericSmallStrainKinematicPlasticity() override
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






protected:



double& GetThreshold() { return mThreshold; }
double& GetPlasticDissipation() { return mPlasticDissipation; }
Vector& GetPlasticStrain() { return mPlasticStrain; }

void SetThreshold(const double Threshold) { mThreshold = Threshold; }
void SetPlasticDissipation(const double PlasticDissipation) { mPlasticDissipation = PlasticDissipation; }
void SetPlasticStrain(const array_1d<double, VoigtSize>& rPlasticStrain) { mPlasticStrain = rPlasticStrain; }

void SetPreviousStressVector(const Vector& toBS) {mPreviousStressVector = toBS; }
Vector& GetPreviousStressVector() { return mPreviousStressVector;}

void SetBackStressVector(const Vector& toBS) {mBackStressVector = toBS; }
Vector& GetBackStressVector() { return mBackStressVector;}





private:


double mPlasticDissipation = 0.0;
double mThreshold = 0.0;
Vector mPlasticStrain = ZeroVector(VoigtSize);

Vector mPreviousStressVector = ZeroVector(VoigtSize);
Vector mBackStressVector     = ZeroVector(VoigtSize);




void CalculateTangentTensor(ConstitutiveLaw::Parameters &rValues);





friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("PlasticDissipation", mPlasticDissipation);
rSerializer.save("Threshold", mThreshold);
rSerializer.save("PlasticStrain", mPlasticStrain);
rSerializer.save("PreviousStressVector", mPreviousStressVector);
rSerializer.save("BackStressVector", mBackStressVector);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("PlasticDissipation", mPlasticDissipation);
rSerializer.load("Threshold", mThreshold);
rSerializer.load("PlasticStrain", mPlasticStrain);
rSerializer.load("PreviousStressVector", mPreviousStressVector);
rSerializer.load("BackStressVector", mBackStressVector);
}


}; 

} 
