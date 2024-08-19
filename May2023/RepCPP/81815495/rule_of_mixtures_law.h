
#pragma once



#include "includes/constitutive_law.h"


namespace Kratos
{





template<unsigned int TDim>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) ParallelRuleOfMixturesLaw
: public ConstitutiveLaw
{
public:


typedef ProcessInfo ProcessInfoType;

typedef ConstitutiveLaw    BaseType;

typedef std::size_t        SizeType;

typedef std::size_t       IndexType;

static constexpr SizeType VoigtSize = (TDim == 3) ? 6 : 3;

static constexpr SizeType Dimension = TDim;

static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( ParallelRuleOfMixturesLaw );



ParallelRuleOfMixturesLaw();


ParallelRuleOfMixturesLaw(const std::vector<double>& rCombinationFactors);


ParallelRuleOfMixturesLaw (const ParallelRuleOfMixturesLaw& rOther);


~ParallelRuleOfMixturesLaw() override;




ConstitutiveLaw::Pointer Clone() const override;


ConstitutiveLaw::Pointer Create(Kratos::Parameters NewParameters) const override;


SizeType WorkingSpaceDimension() override;


SizeType GetStrainSize() const override;


bool RequiresInitializeMaterialResponse() override
{
return true;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


bool Has(const Variable<bool>& rThisVariable) override;


bool Has(const Variable<int>& rThisVariable) override;


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


bool Has(const Variable<Matrix>& rThisVariable) override;


bool Has(const Variable<array_1d<double, 3 > >& rThisVariable) override;


bool Has(const Variable<array_1d<double, 6 > >& rThisVariable) override;


bool& GetValue(
const Variable<bool>& rThisVariable,
bool& rValue
) override;


int& GetValue(
const Variable<int>& rThisVariable,
int& rValue
) override;


double& GetValue(
const Variable<double>& rThisVariable,
double& rValue
) override;


Vector& GetValue(
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


Matrix& GetValue(
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


array_1d<double, 3 > & GetValue(
const Variable<array_1d<double, 3 > >& rThisVariable,
array_1d<double, 3 > & rValue
) override;


array_1d<double, 6 > & GetValue(
const Variable<array_1d<double, 6 > >& rThisVariable,
array_1d<double, 6 > & rValue
) override;


void SetValue(
const Variable<bool>& rThisVariable,
const bool& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<int>& rThisVariable,
const int& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<double>& rThisVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<Vector >& rThisVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<Matrix >& rThisVariable,
const Matrix& rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<array_1d<double, 3 > >& rThisVariable,
const array_1d<double, 3 > & rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValue(
const Variable<array_1d<double, 6 > >& rThisVariable,
const array_1d<double, 6 > & rValue,
const ProcessInfo& rCurrentProcessInfo
) override;


double& CalculateValue(
Parameters& rParameterValues,
const Variable<double>& rThisVariable,
double& rValue
) override;


Vector& CalculateValue(
Parameters& rParameterValues,
const Variable<Vector>& rThisVariable,
Vector& rValue
) override;


Matrix& CalculateValue(
Parameters& rParameterValues,
const Variable<Matrix>& rThisVariable,
Matrix& rValue
) override;


array_1d<double, 3 > & CalculateValue(
Parameters& rParameterValues,
const Variable<array_1d<double, 3 > >& rVariable,
array_1d<double, 3 > & rValue
) override;


array_1d<double, 6 > & CalculateValue(
Parameters& rParameterValues,
const Variable<array_1d<double, 6 > >& rVariable,
array_1d<double, 6 > & rValue
) override;


bool ValidateInput(const Properties& rMaterialProperties) override;


StrainMeasure GetStrainMeasure() override;


StressMeasure GetStressMeasure() override;


bool IsIncremental() override;


void InitializeMaterial(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues
) override;

std::vector<ConstitutiveLaw::Pointer>& GetConstitutiveLaws()
{
return mConstitutiveLaws;
}

std::vector<double>& GetCombinationFactors()
{
return mCombinationFactors;
}

void SetCombinationFactors(const std::vector<double>& rVector )
{
mCombinationFactors = rVector;
}


void CalculateMaterialResponsePK1 (Parameters& rValues) override;


void CalculateMaterialResponsePK2 (Parameters& rValues) override;


void CalculateMaterialResponseKirchhoff (Parameters& rValues) override;


void CalculateMaterialResponseCauchy (Parameters& rValues) override;


void InitializeMaterialResponsePK1 (Parameters& rValues) override;


void InitializeMaterialResponsePK2 (Parameters& rValues) override;


void InitializeMaterialResponseKirchhoff (Parameters& rValues) override;


void InitializeMaterialResponseCauchy (Parameters& rValues) override;


void FinalizeMaterialResponsePK1 (Parameters& rValues) override;


void FinalizeMaterialResponsePK2 (Parameters& rValues) override;


void FinalizeMaterialResponseKirchhoff (Parameters& rValues) override;


void FinalizeMaterialResponseCauchy (Parameters& rValues) override;


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


void CalculateRotationMatrix(
const Properties& rMaterialProperties,
BoundedMatrix<double, VoigtSize, VoigtSize>& rRotationMatrix,
const IndexType Layer
);



void CalculateTangentTensor(
ConstitutiveLaw::Parameters& rValues,
const ConstitutiveLaw::StressMeasure& rStressMeasure
);


void CalculateGreenLagrangeStrain(Parameters &rValues);


void CalculateAlmansiStrain(Parameters &rValues);

protected:






private:



std::vector<ConstitutiveLaw::Pointer> mConstitutiveLaws; 
std::vector<double> mCombinationFactors;                 




friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, ConstitutiveLaw )
rSerializer.save("ConstitutiveLaws", mConstitutiveLaws);
rSerializer.save("CombinationFactors", mCombinationFactors);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, ConstitutiveLaw)
rSerializer.load("ConstitutiveLaws", mConstitutiveLaws);
rSerializer.load("CombinationFactors", mCombinationFactors);
}


}; 
}  
