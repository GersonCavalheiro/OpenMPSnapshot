
#pragma once



#include "includes/constitutive_law.h"


namespace Kratos
{





class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SerialParallelRuleOfMixturesLaw
: public ConstitutiveLaw
{
public:

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(SerialParallelRuleOfMixturesLaw);



SerialParallelRuleOfMixturesLaw()
{
}


SerialParallelRuleOfMixturesLaw(double FiberVolParticipation, const Vector& rParallelDirections)
: mFiberVolumetricParticipation(FiberVolParticipation), mParallelDirections(rParallelDirections)
{
}


ConstitutiveLaw::Pointer Clone() const override
{
return Kratos::make_shared<SerialParallelRuleOfMixturesLaw>(*this);
}

SerialParallelRuleOfMixturesLaw(SerialParallelRuleOfMixturesLaw const& rOther)
: ConstitutiveLaw(rOther), mpMatrixConstitutiveLaw(rOther.mpMatrixConstitutiveLaw), mpFiberConstitutiveLaw(rOther.mpFiberConstitutiveLaw),
mFiberVolumetricParticipation(rOther.mFiberVolumetricParticipation), mParallelDirections(rOther.mParallelDirections) , 
mPreviousStrainVector(rOther.mPreviousStrainVector) , mPreviousSerialStrainMatrix(rOther.mPreviousSerialStrainMatrix) , mIsPrestressed(rOther.mIsPrestressed) 
{
}


~SerialParallelRuleOfMixturesLaw() override
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


bool& GetValue(const Variable<bool>& rThisVariable, bool& rValue) override;


int& GetValue(const Variable<int>& rThisVariable, int& rValue) override;


double& GetValue(const Variable<double>& rThisVariable, double& rValue) override;


Matrix& GetValue(const Variable<Matrix>& rThisVariable, Matrix& rValue) override;


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


bool Has(const Variable<bool>& rThisVariable) override;


bool Has(const Variable<int>& rThisVariable) override;


bool Has(const Variable<double>& rThisVariable) override;


bool Has(const Variable<Vector>& rThisVariable) override;


bool Has(const Variable<Matrix>& rThisVariable) override;


bool& CalculateValue(
Parameters& rParameterValues,
const Variable<bool>& rThisVariable,
bool& rValue) override;


int& CalculateValue(
Parameters& rParameterValues,
const Variable<int>& rThisVariable,
int& rValue) override;


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



void IntegrateStrainSerialParallelBehaviour(
const Vector& rStrainVector,
Vector& FiberStressVector,
Vector& MatrixStressVector,
const Properties& rMaterialProperties,
ConstitutiveLaw::Parameters& rValues,
Vector& rSerialStrainMatrix,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy);


void CalculateSerialParallelProjectionMatrices(
Matrix& rParallelProjector,
Matrix& rSerialProjector);


void InitializeMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;


void InitializeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;



void InitializeMaterialResponsePK1(ConstitutiveLaw::Parameters& rValues) override;



void InitializeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters& rValues) override;


void CalculateGreenLagrangeStrain(Parameters &rValues);


void CalculateAlmansiStrain(Parameters &rValues);


void CalculateStrainsOnEachComponent(
const Vector& rStrainVector,
const Matrix& rParallelProjector,
const Matrix& rSerialProjector,
const Vector& rSerialStrainMatrix,
Vector& rStrainVectorMatrix,
Vector& rStrainVectorFiber,
ConstitutiveLaw::Parameters& rValues,
const int Iteration = 1);


void CalculateInitialApproximationSerialStrainMatrix(
const Vector& rStrainVector,
const Vector& rPreviousStrainVector,
const Properties& rMaterialProperties,
const Matrix& rParallelProjector,
const Matrix& rSerialProjector,
Matrix& rConstitutiveTensorMatrixSS,
Matrix& rConstitutiveTensorFiberSS,
Vector& rInitialApproximationSerialStrainMatrix,
ConstitutiveLaw::Parameters& rValues,
const ConstitutiveLaw::StressMeasure& rStressMeasure);


void IntegrateStressesOfFiberAndMatrix(
ConstitutiveLaw::Parameters& rValues,
Vector& rMatrixStrainVector,
Vector& rFiberStrainVector,
Vector& rMatrixStressVector,
Vector& rFiberStressVector,
const ConstitutiveLaw::StressMeasure& rStressMeasure);


void CheckStressEquilibrium(
ConstitutiveLaw::Parameters& rValues,
const Vector& rStrainVector,
const Matrix& rSerialProjector,
const Vector& rMatrixStressVector,
const Vector& rFiberStressVector,
Vector& rStressResidual,
bool& rIsConverged,
const Matrix& rConstitutiveTensorMatrixSS,
const Matrix& rConstitutiveTensorFiberSS);


void CorrectSerialStrainMatrix(
ConstitutiveLaw::Parameters& rValues,
const Vector& rResidualStresses,
Vector& rSerialStrainMatrix,
const Matrix& rSerialProjector,
const ConstitutiveLaw::StressMeasure& rStressMeasure);


void CalculateTangentTensor(ConstitutiveLaw::Parameters& rValues,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy);


bool RequiresInitializeMaterialResponse() override
{
return true;
}


bool RequiresFinalizeMaterialResponse() override
{
return true;
}


int Check(
const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo
) const override;






protected:




ConstitutiveLaw::Pointer GetMatrixConstitutiveLaw()
{
return mpMatrixConstitutiveLaw;
}


void SetMatrixConstitutiveLaw(ConstitutiveLaw::Pointer pMatrixConstitutiveLaw)
{
mpMatrixConstitutiveLaw = pMatrixConstitutiveLaw;
}


ConstitutiveLaw::Pointer GetFiberConstitutiveLaw()
{
return mpFiberConstitutiveLaw;
}


void SetFiberConstitutiveLaw(ConstitutiveLaw::Pointer pFiberConstitutiveLaw)
{
mpFiberConstitutiveLaw = pFiberConstitutiveLaw;
}


int GetNumberOfSerialComponents()
{
const int parallel_components = inner_prod(mParallelDirections, mParallelDirections);
return this->GetStrainSize() - parallel_components;
}





private:


ConstitutiveLaw::Pointer mpMatrixConstitutiveLaw;
ConstitutiveLaw::Pointer mpFiberConstitutiveLaw;
double mFiberVolumetricParticipation;
array_1d<double, 6> mParallelDirections = ZeroVector(6);
array_1d<double, 6> mPreviousStrainVector = ZeroVector(6);
Vector mPreviousSerialStrainMatrix = ZeroVector(GetNumberOfSerialComponents());
bool mIsPrestressed = false;







friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.save("MatrixConstitutiveLaw", mpMatrixConstitutiveLaw);
rSerializer.save("FiberConstitutiveLaw", mpFiberConstitutiveLaw);
rSerializer.save("FiberVolumetricParticipation", mFiberVolumetricParticipation);
rSerializer.save("ParallelDirections", mParallelDirections);
rSerializer.save("PreviousStrainVector", mPreviousStrainVector);
rSerializer.save("PreviousSerialStrainMatrix", mPreviousSerialStrainMatrix);
rSerializer.save("IsPrestressed", mIsPrestressed);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, ConstitutiveLaw)
rSerializer.load("MatrixConstitutiveLaw", mpMatrixConstitutiveLaw);
rSerializer.load("FiberConstitutiveLaw", mpFiberConstitutiveLaw);
rSerializer.load("FiberVolumetricParticipation", mFiberVolumetricParticipation);
rSerializer.load("ParallelDirections", mParallelDirections);
rSerializer.load("PreviousStrainVector", mPreviousStrainVector);
rSerializer.load("PreviousSerialStrainMatrix", mPreviousSerialStrainMatrix);
rSerializer.load("IsPrestressed", mIsPrestressed);
}


}; 

} 
