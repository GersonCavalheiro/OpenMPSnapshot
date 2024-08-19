
#pragma once



#include "includes/constitutive_law.h"

namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericAnisotropic3DLaw
: public ConstitutiveLaw
{
public:

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

typedef std::size_t SizeType;

static constexpr SizeType Dimension = 3;

static constexpr SizeType VoigtSize = 6;

typedef BoundedMatrix<double, 3, 3> BoundedMatrixType;

typedef BoundedMatrix<double, VoigtSize, VoigtSize> BoundedMatrixVoigtType;

KRATOS_CLASS_POINTER_DEFINITION(GenericAnisotropic3DLaw);



GenericAnisotropic3DLaw()
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<GenericAnisotropic3DLaw>(*this);
}

GenericAnisotropic3DLaw(GenericAnisotropic3DLaw const& rOther)
: ConstitutiveLaw(rOther),
mpIsotropicCL(rOther.mpIsotropicCL)
{
}


~GenericAnisotropic3DLaw() override
{
}




ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;


SizeType WorkingSpaceDimension() override
{
return 3;
};


SizeType GetStrainSize() const override
{
return 6;
};


void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;


Vector& GetValue(const Variable<Vector>& rThisVariable, Vector& rValue) override;


double& GetValue(const Variable<double>& rThisVariable, double& rValue) override;


Matrix& GetValue(const Variable<Matrix>& rThisVariable, Matrix& rValue) override;


bool Has(const Variable<bool>& rThisVariable) override;


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


bool Has(const Variable<Matrix>& rThisVariable) override;


double& CalculateValue(
Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue) override;


Vector& CalculateValue(
Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue) override;


Matrix& CalculateValue(
Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue) override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues) override;



void CalculateAnisotropicStressMapperMatrix(
const Properties& rProperties,
BoundedMatrixVoigtType &rAs,
BoundedMatrixVoigtType& rAsInv
);


void CalculateAnisotropicStrainMapperMatrix(
const BoundedMatrixVoigtType& rAnisotropicElasticMatrix,
const BoundedMatrixVoigtType& rIsotropicElasticMatrix,
const BoundedMatrixVoigtType &rAs,
BoundedMatrixVoigtType& rAe
);


void InitializeMaterialResponsePK2(Parameters& rValues) override;


bool RequiresInitializeMaterialResponse() override
{
return mpIsotropicCL->RequiresInitializeMaterialResponse();
}


bool RequiresFinalizeMaterialResponse() override
{
return mpIsotropicCL->RequiresFinalizeMaterialResponse();
}

void CalculateOrthotropicElasticMatrix(
BoundedMatrixVoigtType &rElasticityTensor,
const Properties &rMaterialProperties);

int Check(const Properties &rMaterialProperties,
const GeometryType &rElementGeometry,
const ProcessInfo &rCurrentProcessInfo) const override;

void CalculateTangentTensor(ConstitutiveLaw::Parameters &rValues);





protected:




ConstitutiveLaw::Pointer GetIsotropicConstitutiveLaw()
{
return mpIsotropicCL;
}


void SetIsotropicConstitutiveLaw(ConstitutiveLaw::Pointer pIsotropicConstitutiveLaw)
{
mpIsotropicCL = pIsotropicConstitutiveLaw;
}






private:


ConstitutiveLaw::Pointer mpIsotropicCL;



void CalculateCauchyGreenStrain(
ConstitutiveLaw::Parameters &rValues,
Vector &rStrainVector);





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("IsotropicCL", mpIsotropicCL);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("IsotropicCL", mpIsotropicCL);
}


}; 

} 
