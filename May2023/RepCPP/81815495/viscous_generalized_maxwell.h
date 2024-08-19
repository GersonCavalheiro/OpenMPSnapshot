
#pragma once



#include "includes/constitutive_law.h"
#include "custom_constitutive/elastic_isotropic_3d.h"

namespace Kratos
{





template<class TElasticBehaviourLaw>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) ViscousGeneralizedMaxwell
: public TElasticBehaviourLaw
{
public:

typedef ConstitutiveLaw CLBaseType;

typedef TElasticBehaviourLaw BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

static constexpr SizeType Dimension = TElasticBehaviourLaw::Dimension;

static constexpr SizeType VoigtSize = TElasticBehaviourLaw::VoigtSize;

KRATOS_CLASS_POINTER_DEFINITION(ViscousGeneralizedMaxwell);

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();



ViscousGeneralizedMaxwell();


ConstitutiveLaw::Pointer Clone() const override;


ViscousGeneralizedMaxwell(const ViscousGeneralizedMaxwell& rOther);


~ViscousGeneralizedMaxwell() override;




void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeSolutionStep(
const Properties &rMaterialProperties,
const GeometryType &rElementGeometry,
const Vector &rShapeFunctionsValues,
const ProcessInfo &rCurrentProcessInfo) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


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


bool RequiresInitializeMaterialResponse() override
{
return false;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}






protected:







private:


Vector mPrevStressVector = ZeroVector(VoigtSize);
Vector mPrevStrainVector = ZeroVector(VoigtSize);



Vector& GetPreviousStressVector() { return mPrevStressVector; }
void SetPreviousStressVector(const Vector& PrevStressVector) { mPrevStressVector = PrevStressVector; }

Vector& GetPreviousStrainVector() { return mPrevStrainVector; }
void SetPreviousStrainVector(const Vector& PrevStrainVector) { mPrevStrainVector = PrevStrainVector; }


void ComputeViscoElasticity(ConstitutiveLaw::Parameters& rValues);





friend class Serializer;

void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("PrevStressVector", mPrevStressVector);
rSerializer.save("PrevStrainVector", mPrevStrainVector);
}

void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("PrevStressVector", mPrevStressVector);
rSerializer.load("PrevStrainVector", mPrevStrainVector);
}


}; 

} 
