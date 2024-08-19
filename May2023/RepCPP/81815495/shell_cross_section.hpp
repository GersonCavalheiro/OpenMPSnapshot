
#pragma once

#include <string>
#include <iostream>

#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/constitutive_law.h"
#include "shell_utilities.h"
#include "containers/flags.h"

namespace Kratos {


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellCrossSection : public Flags
{

public:

class Ply;

KRATOS_CLASS_POINTER_DEFINITION(ShellCrossSection);

typedef Geometry<Node > GeometryType;

typedef std::vector< Ply > PlyCollection;

typedef std::size_t SizeType;



enum SectionBehaviorType {
Thick, 
Thin 
};



struct Features {
Flags mOptions;
double mStrainSize;
double mSpaceDimension;
std::vector< ConstitutiveLaw::StrainMeasure > mStrainMeasures;
};



class SectionParameters
{

private:

Flags                mOptions;

Vector*              mpGeneralizedStrainVector;
Vector*              mpGeneralizedStressVector;
Matrix*              mpConstitutiveMatrix;

double				 mStenbergShearStabilization = 1.0;

const Vector*        mpShapeFunctionsValues;
const Matrix*        mpShapeFunctionsDerivatives;
const ProcessInfo*   mpCurrentProcessInfo;
const Properties*    mpMaterialProperties;
const GeometryType*  mpElementGeometry;

public:

SectionParameters()
: mpGeneralizedStrainVector(nullptr)
, mpGeneralizedStressVector(nullptr)
, mpConstitutiveMatrix(nullptr)
, mpShapeFunctionsValues(nullptr)
, mpShapeFunctionsDerivatives(nullptr)
, mpCurrentProcessInfo(nullptr)
, mpMaterialProperties(nullptr)
, mpElementGeometry(nullptr)
{}

SectionParameters(const GeometryType& rElementGeometry,
const Properties& rMaterialProperties,
const ProcessInfo& rCurrentProcessInfo)
: mpGeneralizedStrainVector(nullptr)
, mpGeneralizedStressVector(nullptr)
, mpConstitutiveMatrix(nullptr)
, mpShapeFunctionsValues(nullptr)
, mpShapeFunctionsDerivatives(nullptr)
, mpCurrentProcessInfo(&rCurrentProcessInfo)
, mpMaterialProperties(&rMaterialProperties)
, mpElementGeometry(&rElementGeometry)
{}

SectionParameters(const SectionParameters& rNewParameters)
: mOptions(rNewParameters.mOptions)
, mpGeneralizedStrainVector(rNewParameters.mpGeneralizedStrainVector)
, mpGeneralizedStressVector(rNewParameters.mpGeneralizedStressVector)
, mpConstitutiveMatrix(rNewParameters.mpConstitutiveMatrix)
, mpShapeFunctionsValues(rNewParameters.mpShapeFunctionsValues)
, mpShapeFunctionsDerivatives(rNewParameters.mpShapeFunctionsDerivatives)
, mpCurrentProcessInfo(rNewParameters.mpCurrentProcessInfo)
, mpMaterialProperties(rNewParameters.mpMaterialProperties)
, mpElementGeometry(rNewParameters.mpElementGeometry)
{}

public:


bool CheckShapeFunctions()
{
if (!mpShapeFunctionsValues) {
KRATOS_THROW_ERROR(std::invalid_argument,"ShapeFunctionsValues NOT SET","");
}

if (!mpShapeFunctionsDerivatives) {
KRATOS_THROW_ERROR(std::invalid_argument,"ShapeFunctionsDerivatives NOT SET","");
}

return 1;
}


bool CheckInfoMaterialGeometry()
{
if (!mpCurrentProcessInfo) {
KRATOS_THROW_ERROR(std::invalid_argument,"CurrentProcessInfo NOT SET","");
}

if (!mpMaterialProperties) {
KRATOS_THROW_ERROR(std::invalid_argument,"MaterialProperties NOT SET","");
}

if (!mpElementGeometry) {
KRATOS_THROW_ERROR(std::invalid_argument,"ElementGeometry NOT SET","");
}

return 1;
}


bool CheckMechanicalVariables()
{
if (!mpGeneralizedStrainVector) {
KRATOS_THROW_ERROR(std::invalid_argument,"GenralizedStrainVector NOT SET","");
}

if (!mpGeneralizedStressVector) {
KRATOS_THROW_ERROR(std::invalid_argument,"GenralizedStressVector NOT SET","");
}

if (!mpConstitutiveMatrix) {
KRATOS_THROW_ERROR(std::invalid_argument,"ConstitutiveMatrix NOT SET","");
}

return 1;
}





void Set(const Flags ThisFlag)
{
mOptions.Set(ThisFlag);
};
void Reset(const Flags ThisFlag)
{
mOptions.Reset(ThisFlag);
};

void SetOptions(const Flags&  rOptions)
{
mOptions=rOptions;
};

void SetGeneralizedStrainVector(Vector& rGeneralizedStrainVector)
{
mpGeneralizedStrainVector=&rGeneralizedStrainVector;
};
void SetGeneralizedStressVector(Vector& rGeneralizedStressVector)
{
mpGeneralizedStressVector=&rGeneralizedStressVector;
};
void SetConstitutiveMatrix(Matrix& rConstitutiveMatrix)
{
mpConstitutiveMatrix =&rConstitutiveMatrix;
};

void SetShapeFunctionsValues(const Vector& rShapeFunctionsValues)
{
mpShapeFunctionsValues=&rShapeFunctionsValues;
};
void SetShapeFunctionsDerivatives(const Matrix& rShapeFunctionsDerivatives)
{
mpShapeFunctionsDerivatives=&rShapeFunctionsDerivatives;
};
void SetProcessInfo(const ProcessInfo& rProcessInfo)
{
mpCurrentProcessInfo =&rProcessInfo;
};
void SetMaterialProperties(const Properties&  rMaterialProperties)
{
mpMaterialProperties =&rMaterialProperties;
};
void SetElementGeometry(const GeometryType& rElementGeometry)
{
mpElementGeometry =&rElementGeometry;
};
void SetStenbergShearStabilization(const double& StenbergShearStabilization)
{
mStenbergShearStabilization = StenbergShearStabilization;
};



Flags& GetOptions()
{
return mOptions;
};

Vector& GetGeneralizedStrainVector()
{
return *mpGeneralizedStrainVector;
};
Vector& GetGeneralizedStressVector()
{
return *mpGeneralizedStressVector;
};
Matrix& GetConstitutiveMatrix()
{
return *mpConstitutiveMatrix;
};

const Vector& GetShapeFunctionsValues()
{
return *mpShapeFunctionsValues;
};
const Matrix& GetShapeFunctionsDerivatives()
{
return *mpShapeFunctionsDerivatives;
};
const ProcessInfo&  GetProcessInfo()
{
return *mpCurrentProcessInfo;
};
const Properties&   GetMaterialProperties()
{
return *mpMaterialProperties;
};
const GeometryType& GetElementGeometry()
{
return *mpElementGeometry;
};
double GetStenbergShearStabilization()
{
return mStenbergShearStabilization;
};
};

class IntegrationPoint
{

private:

double mWeight;
double mLocation;
ConstitutiveLaw::Pointer mConstitutiveLaw;

public:

IntegrationPoint()
: mWeight(0.0)
, mLocation(0.0)
, mConstitutiveLaw(ConstitutiveLaw::Pointer())
{}

IntegrationPoint(double location, double weight, const ConstitutiveLaw::Pointer pMaterial)
: mWeight(weight)
, mLocation(location)
, mConstitutiveLaw(pMaterial)
{}

virtual ~IntegrationPoint() {};

IntegrationPoint(const IntegrationPoint& other)
: mWeight(other.mWeight)
, mLocation(other.mLocation)
, mConstitutiveLaw(other.mConstitutiveLaw != NULL ? other.mConstitutiveLaw->Clone() : ConstitutiveLaw::Pointer())
{}

IntegrationPoint& operator = (const IntegrationPoint& other)
{
if (this != &other) {
mWeight = other.mWeight;
mLocation = other.mLocation;
mConstitutiveLaw = other.mConstitutiveLaw != NULL ? other.mConstitutiveLaw->Clone() : ConstitutiveLaw::Pointer();
}
return *this;
}

public:

inline double GetWeight()const
{
return mWeight;
}
inline void SetWeight(double w)
{
mWeight = w;
}

inline double GetLocation()const
{
return mLocation;
}
inline void SetLocation(double l)
{
mLocation = l;
}

inline const ConstitutiveLaw::Pointer& GetConstitutiveLaw()const
{
return mConstitutiveLaw;
}
inline void SetConstitutiveLaw(const ConstitutiveLaw::Pointer& pLaw)
{
mConstitutiveLaw = pLaw;
}

private:

friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
rSerializer.save("W", mWeight);
rSerializer.save("L", mLocation);
rSerializer.save("CLaw", mConstitutiveLaw);
}

virtual void load(Serializer& rSerializer)
{
rSerializer.load("W", mWeight);
rSerializer.load("L", mLocation);
rSerializer.load("CLaw", mConstitutiveLaw);
}
};

class Ply
{

public:

typedef std::vector< IntegrationPoint > IntegrationPointCollection;

private:

int mPlyIndex;
IntegrationPointCollection mIntegrationPoints;

public:

Ply()
: mPlyIndex(0)
, mIntegrationPoints()
{}

Ply(const int PlyIndex, int NumIntegrationPoints, const Properties& rProps)
: mPlyIndex(PlyIndex)
, mIntegrationPoints()
{
KRATOS_ERROR_IF(NumIntegrationPoints < 1) << "Number of Integration points must be larger than 0!" << std::endl;
if (NumIntegrationPoints < 0) {
NumIntegrationPoints = -NumIntegrationPoints;
}
if (NumIntegrationPoints == 0) {
NumIntegrationPoints = 5;
}
if (NumIntegrationPoints % 2 == 0) {
NumIntegrationPoints += 1;
}
InitializeIntegrationPoints(rProps, NumIntegrationPoints);
}

Ply(const Ply& other)
: mPlyIndex(other.mPlyIndex)
, mIntegrationPoints(other.mIntegrationPoints)
{}

virtual ~Ply() {}

Ply& operator = (const Ply& other)
{
if (this != &other) {
mPlyIndex = other.mPlyIndex;
mIntegrationPoints = other.mIntegrationPoints;
}
return *this;
}

public:

inline double GetThickness(const Properties& rProps) const
{
return ShellUtilities::GetThickness(rProps, mPlyIndex);
}

inline double GetLocation(const Properties& rProps) const
{
double my_location(0.0);

double current_location = ShellUtilities::GetThickness(rProps) * 0.5;
const double offset = GetOffset(rProps);

for (int i=0; i<mPlyIndex+1; ++i) {
double ply_thickness = GetThickness(rProps);
my_location = current_location - ply_thickness*0.5 - offset;
current_location -= ply_thickness;
}
return my_location;
}


inline double GetOrientationAngle(const Properties& rProps) const
{
return ShellUtilities::GetOrientationAngle(rProps, mPlyIndex);
}

inline double GetOffset(const Properties& rProps) const
{
return ShellUtilities::GetOffset(rProps);
}

void RecoverOrthotropicProperties(const IndexType currentPly, Properties& laminaProps);

inline IntegrationPointCollection& GetIntegrationPoints(const Properties& rProps)
{
UpdateIntegrationPoints(rProps);
return mIntegrationPoints;
}

inline double CalculateMassPerUnitArea(const Properties& rProps) const
{
return ShellUtilities::GetDensity(rProps, mPlyIndex) * GetThickness(rProps);
}

inline IntegrationPointCollection::size_type NumberOfIntegrationPoints() const
{
return mIntegrationPoints.size();
}

inline void SetConstitutiveLawAt(IntegrationPointCollection::size_type integrationPointID, const ConstitutiveLaw::Pointer& pNewConstitutiveLaw)
{
if (integrationPointID < mIntegrationPoints.size()) {
mIntegrationPoints[integrationPointID].SetConstitutiveLaw(pNewConstitutiveLaw);
}
}

private:

void InitializeIntegrationPoints(const Properties& rProps, const int NumIntegrationPoints)
{
KRATOS_TRY

const ConstitutiveLaw::Pointer& pMaterial = rProps[CONSTITUTIVE_LAW];
KRATOS_ERROR_IF(pMaterial == nullptr) << "A Ply needs a constitutive law to be set. "
<< "Missing constitutive law in property: " <<  rProps.Id() << std::endl;;

mIntegrationPoints.clear();
mIntegrationPoints.resize(NumIntegrationPoints);
for (int i=0; i<NumIntegrationPoints; ++i) {
mIntegrationPoints[i].SetConstitutiveLaw(pMaterial->Clone());
}

KRATOS_CATCH("")
}
void UpdateIntegrationPoints(const Properties& rProps)
{
KRATOS_TRY

const SizeType num_int_points = mIntegrationPoints.size();

Vector ip_w(num_int_points, 1.0);
if (num_int_points >= 3) {
for (IndexType i=1; i<num_int_points-1; ++i) {
double iw = (i % 2 == 0) ? 2.0 : 4.0;
ip_w(i) = iw;
}
ip_w /= sum(ip_w);
}

const double location = GetLocation(rProps);
const double thickness = GetThickness(rProps);

Vector ip_loc(num_int_points, 0.0);
if (num_int_points >= 3) {
double loc_start = location + 0.5 * thickness;
double loc_incr = thickness / double(num_int_points-1);
for (IndexType i=0; i<num_int_points; ++i) {
ip_loc(i) = loc_start;
loc_start -= loc_incr;
}
}

for (IndexType i=0; i<num_int_points; ++i) {
IntegrationPoint& r_int_point = mIntegrationPoints[i];
r_int_point.SetWeight(ip_w(i) * thickness);
r_int_point.SetLocation(ip_loc(i));
}

KRATOS_CATCH("")
}

private:

friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
rSerializer.save("idx", mPlyIndex);
rSerializer.save("IntP", mIntegrationPoints);
}

virtual void load(Serializer& rSerializer)
{
rSerializer.load("idx", mPlyIndex);
rSerializer.load("IntP", mIntegrationPoints);
}

};

protected:

struct GeneralVariables {
double DeterminantF;
double DeterminantF0;

Vector StrainVector_2D;
Vector StressVector_2D;
Matrix ConstitutiveMatrix_2D;
Matrix DeformationGradientF_2D;
Matrix DeformationGradientF0_2D;

Vector StrainVector_3D;
Vector StressVector_3D;
Matrix ConstitutiveMatrix_3D;
Matrix DeformationGradientF_3D;
Matrix DeformationGradientF0_3D;

double GYZ;
double GXZ;

Matrix H;
Matrix L;
Matrix LT;
Vector CondensedStressVector;
};


public:



ShellCrossSection();


ShellCrossSection(const ShellCrossSection& other);


~ShellCrossSection() override;




ShellCrossSection& operator = (const ShellCrossSection& other);




void BeginStack();


void AddPly(const IndexType PlyIndex, int numPoints, const Properties& rProps);


void EndStack();


virtual std::string GetInfo(const Properties& rProps);


virtual ShellCrossSection::Pointer Clone()const;


virtual bool Has(const Variable<double>& rThisVariable);


virtual bool Has(const Variable<Vector>& rThisVariable);


virtual bool Has(const Variable<Matrix>& rThisVariable);


virtual bool Has(const Variable<array_1d<double, 3 > >& rThisVariable);


virtual bool Has(const Variable<array_1d<double, 6 > >& rThisVariable);


virtual double& GetValue(const Variable<double>& rThisVariable, const Properties& rProps, double& rValue);


virtual Vector& GetValue(const Variable<Vector>& rThisVariable, Vector& rValue);


virtual Matrix& GetValue(const Variable<Matrix>& rThisVariable, Matrix& rValue);


virtual array_1d<double, 3 >& GetValue(const Variable<array_1d<double, 3 > >& rVariable,
array_1d<double, 3 >& rValue);


virtual array_1d<double, 6 >& GetValue(const Variable<array_1d<double, 6 > >& rVariable,
array_1d<double, 6 >& rValue);


virtual void SetValue(const Variable<double>& rVariable,
const double& rValue,
const ProcessInfo& rCurrentProcessInfo);


virtual void SetValue(const Variable<Vector >& rVariable,
const Vector& rValue,
const ProcessInfo& rCurrentProcessInfo);


virtual void SetValue(const Variable<Matrix >& rVariable,
const Matrix& rValue,
const ProcessInfo& rCurrentProcessInfo);


virtual void SetValue(const Variable<array_1d<double, 3 > >& rVariable,
const array_1d<double, 3 >& rValue,
const ProcessInfo& rCurrentProcessInfo);


virtual void SetValue(const Variable<array_1d<double, 6 > >& rVariable,
const array_1d<double, 6 >& rValue,
const ProcessInfo& rCurrentProcessInfo);


virtual bool ValidateInput(const Properties& rMaterialProperties);


virtual void InitializeCrossSection(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues);


virtual void InitializeSolutionStep(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo);


virtual void FinalizeSolutionStep(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo);


virtual void InitializeNonLinearIteration(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo);


virtual void FinalizeNonLinearIteration(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues,
const ProcessInfo& rCurrentProcessInfo);


virtual void CalculateSectionResponse(SectionParameters& rValues, const ConstitutiveLaw::StressMeasure& rStressMeasure);


virtual void FinalizeSectionResponse(SectionParameters& rValues, const ConstitutiveLaw::StressMeasure& rStressMeasure);


virtual void ResetCrossSection(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues);


virtual int Check(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const ProcessInfo& rCurrentProcessInfo);


inline void GetRotationMatrixForGeneralizedStrains(double radians, Matrix& T)
{
double c = std::cos(radians);
double s = std::sin(radians);

SizeType strain_size = GetStrainSize();

if (T.size1() != strain_size || T.size2() != strain_size) {
T.resize(strain_size, strain_size, false);
}
noalias(T) = ZeroMatrix(strain_size, strain_size);

T(0, 0) = c * c;
T(0, 1) =   s * s;
T(0, 2) = - s * c;
T(1, 0) = s * s;
T(1, 1) =   c * c;
T(1, 2) =   s * c;
T(2, 0) = 2.0 * s * c;
T(2, 1) = - 2.0 * s * c;
T(2, 2) = c * c - s * s;

project(T, range(3, 6), range(3, 6)) = project(T, range(0, 3), range(0, 3));

if (strain_size == 8) {
T(6, 6) =   c;
T(6, 7) = s;
T(7, 6) = - s;
T(7, 7) = c;
}
}


inline void GetRotationMatrixForCondensedStrains(double radians, Matrix& T)
{
SizeType strain_size = GetCondensedStrainSize();

if (T.size1() != strain_size || T.size2() != strain_size) {
T.resize(strain_size, strain_size, false);
}
noalias(T) = ZeroMatrix(strain_size, strain_size);

T(0, 0) = 1.0; 

if (strain_size == 3) { 
double c = std::cos(radians);
double s = std::sin(radians);

T(1, 1) =   c;
T(1, 2) = s;
T(2, 1) = - s;
T(2, 2) = c;
}
}


inline void GetRotationMatrixForGeneralizedStresses(double radians, Matrix& T)
{
double c = std::cos(radians);
double s = std::sin(radians);

SizeType strain_size = GetStrainSize();

if (T.size1() != strain_size || T.size2() != strain_size) {
T.resize(strain_size, strain_size, false);
}
noalias(T) = ZeroMatrix(strain_size, strain_size);

T(0, 0) = c * c;
T(0, 1) =   s * s;
T(0, 2) = - 2.0 * s * c;
T(1, 0) = s * s;
T(1, 1) =   c * c;
T(1, 2) =   2.0 * s * c;
T(2, 0) = s * c;
T(2, 1) = - s * c;
T(2, 2) = c * c - s * s;

project(T, range(3, 6), range(3, 6)) = project(T, range(0, 3), range(0, 3));

if (strain_size == 8) {
T(6, 6) =   c;
T(6, 7) = s;
T(7, 6) = - s;
T(7, 7) = c;
}
}


inline void GetRotationMatrixForCondensedStresses(double radians, Matrix& T)
{
SizeType strain_size = GetCondensedStrainSize();

if (T.size1() != strain_size || T.size2() != strain_size) {
T.resize(strain_size, strain_size, false);
}
noalias(T) = ZeroMatrix(strain_size, strain_size);

T(0, 0) = 1.0; 

if (strain_size == 3) { 
double c = std::cos(radians);
double s = std::sin(radians);

T(1, 1) =   c;
T(1, 2) = s;
T(2, 1) = - s;
T(2, 2) = c;
}
}


public:



inline double GetThickness(const Properties& rProps) const
{
double thickness = 0.0;
for (const auto& r_ply : mStack) {
thickness += r_ply.GetThickness(rProps);
}
return thickness;
}


inline double GetOffset(const Properties& rProps) const
{
KRATOS_DEBUG_ERROR_IF(mStack.size() == 0) << "no plies available!" << std::endl;
return mStack[0].GetOffset(rProps);
}


void GetPlyThicknesses(const Properties& rProps, Vector& rPlyThicknesses)
{
KRATOS_DEBUG_ERROR_IF_NOT(mStack.size() == rPlyThicknesses.size()) << "Size mismatch!" << std::endl;
for (IndexType i_ply=0; i_ply<mStack.size(); ++i_ply) {
rPlyThicknesses[i_ply] = mStack[i_ply].GetThickness(rProps);
}
}


void SetupGetPlyConstitutiveMatrices()
{
mStorePlyConstitutiveMatrices = true;
mPlyConstitutiveMatrices = std::vector<Matrix>(this->NumberOfPlies());

for (IndexType ply = 0; ply < this->NumberOfPlies(); ++ply) {
if (mBehavior == Thick) {
mPlyConstitutiveMatrices[ply].resize(8, 8, false);
} else {
mPlyConstitutiveMatrices[ply].resize(6, 6, false);
}

mPlyConstitutiveMatrices[ply].clear();
}
}


Matrix GetPlyConstitutiveMatrix(const IndexType PlyIndex)
{
return mPlyConstitutiveMatrices[PlyIndex];
}


inline SizeType NumberOfPlies() const
{
return mStack.size();
}


inline SizeType NumberOfIntegrationPointsAt(const IndexType PlyIndex) const
{
if (PlyIndex < mStack.size()) {
return mStack[PlyIndex].NumberOfIntegrationPoints();
}
return 0;
}


inline void SetConstitutiveLawAt(const IndexType PlyIndex, SizeType point_id, const ConstitutiveLaw::Pointer& pNewConstitutiveLaw)
{
if (PlyIndex < mStack.size()) {
mStack[PlyIndex].SetConstitutiveLawAt(point_id, pNewConstitutiveLaw);
}
}


inline double CalculateMassPerUnitArea(const Properties& rProps) const
{
double vol(0.0);
for (const auto& r_ply : mStack) {
vol += r_ply.CalculateMassPerUnitArea(rProps);
}
return vol;
}


inline double CalculateAvarageDensity(const Properties& rProps) const
{
return CalculateMassPerUnitArea(rProps) / GetThickness(rProps);
}


inline double GetOrientationAngle() const
{
return mOrientation;
}


inline void SetOrientationAngle(const double Radians)
{
mOrientation = Radians;
}


inline SectionBehaviorType GetSectionBehavior() const
{
return mBehavior;
}


inline void SetSectionBehavior(SectionBehaviorType behavior)
{
mBehavior = behavior;
}


inline SizeType GetStrainSize()
{
return (mBehavior == Thick) ? 8 : 6;
}


inline SizeType GetCondensedStrainSize()
{
return (mBehavior == Thick) ? 1 : 3;
}


inline double GetDrillingStiffness() const
{
return mDrillingPenalty;
}

std::vector<ConstitutiveLaw::Pointer> GetConstitutiveLawsVector(const Properties& rProps);


void ParseOrthotropicPropertyMatrix(const Properties& pProps);


void GetLaminaeOrientation(const Properties& pProps, Vector& rOrientation_Vector);


void GetLaminaeStrengths(std::vector<Matrix>& rLamina_Strengths, const Properties& rProps);

private:


void InitializeParameters(SectionParameters& rValues, ConstitutiveLaw::Parameters& rMaterialValues, GeneralVariables& rVariables);

void UpdateIntegrationPointParameters(const IntegrationPoint& rPoint, ConstitutiveLaw::Parameters& rMaterialValues, GeneralVariables& rVariables);

void CalculateIntegrationPointResponse(const IntegrationPoint& rPoint,
ConstitutiveLaw::Parameters& rMaterialValues,
SectionParameters& rValues,
GeneralVariables& rVariables,
const ConstitutiveLaw::StressMeasure& rStressMeasure,
const unsigned int& plyNumber);


void PrivateCopy(const ShellCrossSection& other);


public:



private:


PlyCollection mStack;
bool mEditingStack;
bool mHasDrillingPenalty;
double mDrillingPenalty;
double mOrientation;
SectionBehaviorType mBehavior;
bool mInitialized;
bool mNeedsOOPCondensation;
Vector mOOP_CondensedStrains;
Vector mOOP_CondensedStrains_converged;
bool mStorePlyConstitutiveMatrices = false;
std::vector<Matrix> mPlyConstitutiveMatrices;



friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Flags);
rSerializer.save("stack", mStack);
rSerializer.save("edit", mEditingStack);
rSerializer.save("dr", mHasDrillingPenalty);
rSerializer.save("bdr", mDrillingPenalty);
rSerializer.save("or", mOrientation);

rSerializer.save("behav", (int)mBehavior);

rSerializer.save("init", mInitialized);
rSerializer.save("hasOOP", mNeedsOOPCondensation);
rSerializer.save("OOP_eps", mOOP_CondensedStrains);
rSerializer.save("OOP_eps_conv", mOOP_CondensedStrains_converged);
rSerializer.save("store_ply_mat", mStorePlyConstitutiveMatrices);
rSerializer.save("ply_mat", mPlyConstitutiveMatrices);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Flags);
rSerializer.load("stack", mStack);
rSerializer.load("edit", mEditingStack);
rSerializer.load("dr", mHasDrillingPenalty);
rSerializer.load("bdr", mDrillingPenalty);
rSerializer.load("or", mOrientation);

int temp;
rSerializer.load("behav", temp);
mBehavior = (SectionBehaviorType)temp;

rSerializer.load("init", mInitialized);
rSerializer.load("hasOOP", mNeedsOOPCondensation);
rSerializer.load("OOP_eps", mOOP_CondensedStrains);
rSerializer.load("OOP_eps_conv", mOOP_CondensedStrains_converged);
rSerializer.load("store_ply_mat", mStorePlyConstitutiveMatrices);
rSerializer.load("ply_mat", mPlyConstitutiveMatrices);
}


};


inline std::istream& operator >> (std::istream& rIStream, ShellCrossSection& rThis);

inline std::ostream& operator << (std::ostream& rOStream, ShellCrossSection& rThis)
{
return rOStream; 
}


}
